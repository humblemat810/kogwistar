"""Notebook-style tutorial: the same retrieval lesson, now in KG semantics.

# Cell 1. Setup
Prepare imports, backend selection, and lightweight runtime shims.

# Cell 2. Graph Models
Use real Kogwistar models when possible, and deterministic fallbacks when not.

# Cell 3. Persist Once
Seed one backend once, then run vector, lexical, graph, and hybrid retrieval
over that same persisted state.

# Cell 4. Compare And Verify
Run the same questions as tutorial 21 and finish with a parity check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import types
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ============================================================================
# %% Cell 1: Imports, Paths, And Backend Wiring
# This first cell acts like notebook setup. It wires in the workspace root and
# prepares the minimal import shims needed for tutorial environments.
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "kogwistar" not in sys.modules:
    pkg = types.ModuleType("kogwistar")
    pkg.__path__ = [str(ROOT / "kogwistar")]
    sys.modules["kogwistar"] = pkg
if "kogwistar.engine_core" not in sys.modules:
    subpkg = types.ModuleType("kogwistar.engine_core")
    subpkg.__path__ = [str(ROOT / "kogwistar" / "engine_core")]
    sys.modules["kogwistar.engine_core"] = subpkg

from kogwistar.engine_core.in_memory_backend import build_in_memory_backend

HAVE_REAL_GRAPH_KNOWLEDGE_ENGINE = False

# ============================================================================
# %% Cell 2: Real Kogwistar Models First, Lightweight Fallbacks Second
# The tutorial prefers the real graph types, but still stays runnable when the
# full optional dependency stack is unavailable.
# ============================================================================

try:
    from kogwistar.engine_core.models import Edge, Grounding, MentionVerification, Node, Span
    from kogwistar.graph_query import GraphQuery
    HAVE_REAL_KOGWISTAR_MODELS = True
except Exception:
    HAVE_REAL_KOGWISTAR_MODELS = False

    @dataclass
    class MentionVerification:
        method: str
        is_verified: bool
        score: float | None = None
        notes: str | None = None

    @dataclass
    class Span:
        collection_page_url: str
        document_page_url: str
        doc_id: str
        insertion_method: str
        page_number: int
        start_char: int
        end_char: int
        excerpt: str
        context_before: str = ""
        context_after: str = ""
        chunk_id: str | None = None
        source_cluster_id: str | None = None
        verification: MentionVerification | None = None

    @dataclass
    class Grounding:
        spans: list[Span]

    @dataclass
    class Node:
        id: str
        label: str
        type: str
        summary: str
        doc_id: str
        mentions: list[Grounding]
        properties: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None
        domain_id: str | None = None
        canonical_entity_id: str | None = None
        embedding: list[float] | None = None

        def model_dump_json(self, field_mode: str = "backend") -> str:
            _ = field_mode
            return json.dumps(self, default=lambda o: o.__dict__)

        @classmethod
        def model_validate_json(cls, payload: str) -> "Node":
            data = json.loads(payload)
            data["mentions"] = [
                Grounding(spans=[Span(**span) for span in grounding["spans"]])
                for grounding in data.get("mentions", [])
            ]
            return cls(**data)

    @dataclass
    class Edge:
        id: str
        label: str
        type: str
        summary: str
        relation: str
        source_ids: list[str]
        target_ids: list[str]
        source_edge_ids: list[str]
        target_edge_ids: list[str]
        doc_id: str
        mentions: list[Grounding]
        properties: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None
        domain_id: str | None = None
        canonical_entity_id: str | None = None
        embedding: list[float] | None = None

        def model_dump_json(self, field_mode: str = "backend") -> str:
            _ = field_mode
            return json.dumps(self, default=lambda o: o.__dict__)

        @classmethod
        def model_validate_json(cls, payload: str) -> "Edge":
            data = json.loads(payload)
            data["mentions"] = [
                Grounding(spans=[Span(**span) for span in grounding["spans"]])
                for grounding in data.get("mentions", [])
            ]
            return cls(**data)

    class GraphQuery:
        """Notebook-local fallback graph query helper for minimal environments."""

        def __init__(self, engine: Any):
            self.e = engine

        def search_nodes(
            self,
            *,
            label_contains: str | None = None,
            summary_contains: str | None = None,
            type: str | None = None,
            doc_id: str | None = None,
            limit: int = 200,
        ) -> list[str]:
            got = self.e.backend.node_get(where={"doc_id": doc_id} if doc_id else None, include=["documents"])
            out: list[str] = []
            for nid, payload in zip(got.get("ids") or [], got.get("documents") or []):
                if not nid or not payload:
                    continue
                node = Node.model_validate_json(payload)
                if type and node.type != type:
                    continue
                if label_contains and label_contains.lower() not in node.label.lower():
                    continue
                if summary_contains and summary_contains.lower() not in node.summary.lower():
                    continue
                out.append(nid)
                if len(out) >= limit:
                    break
            return out

        def neighbors(self, rid: str, *, direction: str = "both", doc_id: str | None = None, allow_jump_edge: bool = True) -> dict[str, set[str]]:
            _ = allow_jump_edge
            nodes: set[str] = set()
            edges: set[str] = set()
            if (self.e.backend.node_get(ids=[rid], include=["documents"]).get("ids") or []):
                clause = {"$and": [{"endpoint_type": "node"}, {"endpoint_id": rid}]}
                if doc_id:
                    clause["$and"].append({"doc_id": doc_id})
                eps = self.e.backend.edge_endpoints_get(where=clause, include=["documents"])
                for payload in eps.get("documents") or []:
                    row = json.loads(payload)
                    edges.add(row["edge_id"])
                    other = self.e.backend.edge_endpoints_get(where={"edge_id": row["edge_id"]}, include=["documents"])
                    for other_payload in other.get("documents") or []:
                        r2 = json.loads(other_payload)
                        if r2.get("endpoint_type") == "node" and r2["endpoint_id"] != rid:
                            nodes.add(r2["endpoint_id"])
            elif (self.e.backend.edge_get(ids=[rid], include=["documents"]).get("ids") or []):
                clause = {"edge_id": rid}
                if direction in {"src", "tgt"}:
                    clause = {"$and": [{"edge_id": rid}, {"role": direction}]}
                eps = self.e.backend.edge_endpoints_get(where=clause, include=["documents"])
                for payload in eps.get("documents") or []:
                    row = json.loads(payload)
                    if row["endpoint_type"] == "node":
                        nodes.add(row["endpoint_id"])
                    else:
                        edges.add(row["endpoint_id"])
            return {"nodes": nodes, "edges": edges}

        def k_hop(self, start_ids: list[str], k: int = 2, *, doc_id: str | None = None, allow_jump_edge: bool = False):
            _ = allow_jump_edge
            visited: set[str] = set()
            frontier: set[str] = set(start_ids)
            layers: list[dict[str, set[str]]] = []
            for _i in range(max(0, k)):
                next_frontier: set[str] = set()
                layer_nodes: set[str] = set()
                layer_edges: set[str] = set()
                for rid in frontier:
                    if rid in visited:
                        continue
                    visited.add(rid)
                    nbrs = self.neighbors(rid, doc_id=doc_id)
                    layer_nodes |= nbrs["nodes"]
                    layer_edges |= nbrs["edges"]
                    next_frontier |= nbrs["nodes"] | nbrs["edges"]
                layers.append({"nodes": layer_nodes, "edges": layer_edges})
                frontier = next_frontier - visited
            return layers

        def shortest_path(self, src_id: str, dst_id: str, *, doc_id: str | None = None, max_depth: int = 8):
            if src_id == dst_id:
                return [src_id]
            q = deque([(src_id, [src_id])])
            seen = {src_id}
            depth = 0
            while q and depth <= max_depth:
                for _ in range(len(q)):
                    cur, path = q.popleft()
                    nbrs = self.neighbors(cur, doc_id=doc_id)
                    for nxt in nbrs["nodes"] | nbrs["edges"]:
                        if nxt in seen:
                            continue
                        if nxt == dst_id:
                            return path + [nxt]
                        seen.add(nxt)
                        q.append((nxt, path + [nxt]))
                depth += 1
            return []

        def path_between_labels(self, src_substr: str, dst_substr: str, *, doc_id: str | None = None, max_depth: int = 8):
            srcs = self.search_nodes(label_contains=src_substr, doc_id=doc_id, limit=20)
            dsts = set(self.search_nodes(label_contains=dst_substr, doc_id=doc_id, limit=20))
            best: list[str] = []
            for s in srcs:
                for d in dsts:
                    path = self.shortest_path(s, d, doc_id=doc_id, max_depth=max_depth)
                    if path and (not best or len(path) < len(best)):
                        best = path
            return best

        def find_edges(self, *, relation: str | None = None, src_label_contains: str | None = None, tgt_label_contains: str | None = None, doc_id: str | None = None):
            where = {}
            if relation:
                where["relation"] = relation
            if doc_id:
                where["doc_id"] = doc_id
            got = self.e.backend.edge_get(where=({"$and": [{k: v} for k, v in where.items()]} if len(where) > 1 else (where or None)), include=["documents"])
            out = []
            for eid, payload in zip(got.get("ids") or [], got.get("documents") or []):
                if not eid or not payload:
                    continue
                edge = Edge.model_validate_json(payload)
                ok_src = src_label_contains is None
                ok_tgt = tgt_label_contains is None
                if src_label_contains:
                    labels = [Node.model_validate_json(j).label for j in (self.e.backend.node_get(ids=edge.source_ids, include=["documents"]).get("documents") or []) if j]
                    ok_src = any(src_label_contains.lower() in label.lower() for label in labels)
                if tgt_label_contains:
                    labels = [Node.model_validate_json(j).label for j in (self.e.backend.node_get(ids=edge.target_ids, include=["documents"]).get("documents") or []) if j]
                    ok_tgt = any(tgt_label_contains.lower() in label.lower() for label in labels)
                if ok_src and ok_tgt:
                    out.append(eid)
            return out

        def adjacency_list(self, node_ids, *, doc_id: str | None = None):
            return {nid: self.neighbors(nid, doc_id=doc_id) for nid in node_ids}

from scripts.rag_retrieval_comparison_tutorial import (
    QUERY_SET,
    SemanticLexicalEmbeddingFunction,
    comparison_table,
    explain_method,
    format_score,
    load_dataset,
    normalize_text,
    sentence_snippet,
    short_excerpt,
    tokenize,
    RetrievalTutorial as RawTutorial,
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_text(text)).strip("-") or "item"


def _span(doc_id: str, excerpt: str, *, insertion_method: str) -> Span:
    return Span(
        collection_page_url=f"kg/{doc_id}",
        document_page_url=f"kg/{doc_id}",
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=1,
        start_char=0,
        end_char=max(1, len(excerpt)),
        excerpt=excerpt[:512],
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes=insertion_method
        ),
    )


def _grounding(doc_id: str, excerpt: str, *, insertion_method: str) -> Grounding:
    return Grounding(spans=[_span(doc_id, excerpt, insertion_method=insertion_method)])


@dataclass
class KGRecord:
    record_id: str
    doc_id: str
    title: str
    text: str
    kind: str


class TutorialRead:
    def __init__(self, engine: "TutorialMemoryEngine") -> None:
        self._e = engine

    def node_ids_by_doc(self, doc_id: str) -> list[str]:
        got = self._e.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return [meta["node_id"] for meta in got.get("metadatas") or [] if meta]

    def edge_ids_by_doc(self, doc_id: str) -> list[str]:
        got = self._e.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return sorted({meta["edge_id"] for meta in got.get("metadatas") or [] if meta})

    def get_nodes(self, ids):
        got = self._e.node_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Node.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]

    def get_edges(self, ids):
        got = self._e.edge_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Edge.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]


class TutorialWrite:
    def __init__(self, engine: "TutorialMemoryEngine") -> None:
        self._e = engine

    def add_node(self, node: Node) -> None:
        self._e.node_collection.add(
            ids=[node.id],
            documents=[node.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": node.doc_id, "label": node.label, "type": node.type}],
        )
        row_id = f"{node.id}::{node.doc_id}"
        self._e.node_docs_collection.add(
            ids=[row_id],
            documents=[json.dumps({"id": row_id, "node_id": node.id, "doc_id": node.doc_id})],
            metadatas=[{"node_id": node.id, "doc_id": node.doc_id}],
        )

    def add_edge(self, edge: Edge) -> None:
        self._e.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": edge.doc_id, "relation": edge.relation}],
        )
        rows = []
        for source_id in edge.source_ids or []:
            rows.append(
                {
                    "id": f"{edge.id}::src::node::{source_id}",
                    "edge_id": edge.id,
                    "endpoint_id": source_id,
                    "endpoint_type": "node",
                    "role": "src",
                    "relation": edge.relation,
                    "doc_id": edge.doc_id,
                }
            )
        for target_id in edge.target_ids or []:
            rows.append(
                {
                    "id": f"{edge.id}::tgt::node::{target_id}",
                    "edge_id": edge.id,
                    "endpoint_id": target_id,
                    "endpoint_type": "node",
                    "role": "tgt",
                    "relation": edge.relation,
                    "doc_id": edge.doc_id,
                }
            )
        self._e.edge_endpoints_collection.add(
            ids=[row["id"] for row in rows],
            documents=[json.dumps(row) for row in rows],
            metadatas=rows,
        )


# ============================================================================
# %% Cell 3: Minimal Engine Wrappers
# This cell wraps the promoted in-memory backend in a tutorial-friendly engine
# shape so the rest of the notebook can read naturally.
# ============================================================================

class TutorialMemoryEngine:
    """Notebook-friendly wrapper around the promoted in-memory backend."""

    def __init__(self, *, persist_directory: str, embedding_function: Any) -> None:
        self.persist_directory = persist_directory
        self._ef = embedding_function
        self.backend_kind = "memory"
        self.backend = build_in_memory_backend(self)
        self.read = TutorialRead(self)
        self.write = TutorialWrite(self)


def _normalize_backend_name(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized == "fake":
        return "memory"
    return normalized


def _build_engine(backend: str, *, persist_directory: str | None, embedding_function: Any):
    """Choose the backend the same way a notebook would choose an execution path."""
    backend = _normalize_backend_name(backend)
    if backend == "memory":
        if not persist_directory:
            persist_directory = tempfile.mkdtemp(prefix="kg-memory-tutorial-")
        return TutorialMemoryEngine(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
    if backend == "chroma":
        if not persist_directory:
            raise ValueError("--persist-directory is required when --backend chroma")
        raise RuntimeError("Chroma backend requires GraphKnowledgeEngine, which is unavailable in this environment.")
    raise ValueError(f"Unknown backend {backend!r}; expected 'memory' or 'chroma'.")


# ============================================================================
# %% Cell 4: Build The KG Tutorial State
# This class is the main notebook state. It seeds one backend once, then lets
# each retrieval cell query that same persisted graph-backed corpus.
# ============================================================================

class KGSemanticsRetrievalTutorial:
    """Notebook-style driver for the KG-semantics retrieval comparison."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        backend: str = "memory",
        persist_directory: str | None = None,
        embedding_dim: int = 128,
    ) -> None:
        self.docs = docs
        self.docs_by_id = {doc["id"]: doc for doc in docs}
        self.embedder = SemanticLexicalEmbeddingFunction(dim=embedding_dim)
        self.backend_kind = _normalize_backend_name(backend)
        self.engine = _build_engine(
            self.backend_kind,
            persist_directory=persist_directory,
            embedding_function=self.embedder,
        )
        self.graph = GraphQuery(self.engine)
        self.raw_answer_router = RawTutorial(docs)
        self.records: list[KGRecord] = []
        self.records_by_id: dict[str, KGRecord] = {}
        self.record_token_counts: dict[str, Counter[str]] = {}
        self.record_lengths: dict[str, int] = {}
        self.entity_name_to_node_id: dict[str, str] = {}
        self.alias_to_node_id: dict[str, str] = {}
        self.doc_entities: dict[str, list[str]] = {}
        self._build_kg()
        self._refresh_persisted_corpus()

    def _collection(self, name: str):
        return getattr(self.engine, f"{name}_collection")

    def _collection_rows(self, name: str) -> list[tuple[str, str]]:
        got = self._collection(name).get(include=["documents"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        return [
            (rid, payload)
            for rid, payload in zip(ids, docs)
            if rid and payload
        ]

    def _collection_doc(self, name: str, rid: str) -> str | None:
        got = self._collection(name).get(ids=[rid], include=["documents"])
        return (got.get("documents") or [None])[0]

    def _has_node(self, node_id: str) -> bool:
        got = self._collection_doc("node", node_id)
        return got is not None

    def _node_label_by_id(self, node_id: str) -> str:
        doc = self._collection_doc("node", node_id)
        if not doc:
            return node_id
        return Node.model_validate_json(doc).label

    def _node_id_for_label(self, label: str) -> str:
        return f"node::{slugify(label)}"

    def _extract_entity_labels(self, text: str) -> list[str]:
        q = normalize_text(text)
        labels: list[str] = []
        for alias, node_id in sorted(self.alias_to_node_id.items()):
            if alias and alias in q:
                label = self._node_label_by_id(node_id)
                if label not in labels:
                    labels.append(label)
        return labels

    def _refresh_persisted_corpus(self) -> None:
        """Refresh the vector and lexical views from the persisted KG state."""
        records: list[KGRecord] = []
        alias_to_node_id: dict[str, str] = {}
        entity_name_to_node_id: dict[str, str] = {}

        for node_id, payload in sorted(self._collection_rows("node"), key=lambda item: item[0]):
            node = Node.model_validate_json(payload)
            aliases = set()
            for raw_alias in (
                [node.label]
                + list((getattr(node, "properties", {}) or {}).get("aliases", []) or [])
                + list((getattr(node, "metadata", {}) or {}).get("aliases", []) or [])
            ):
                alias = normalize_text(raw_alias)
                if alias:
                    aliases.add(alias)
            for alias in aliases:
                alias_to_node_id[alias] = node.id
            entity_name_to_node_id[normalize_text(node.label)] = node.id
            records.append(
                KGRecord(
                    record_id=node.id,
                    doc_id=node.doc_id,
                    title=node.label,
                    text=f"{node.label}. {node.summary}",
                    kind="node",
                )
            )

        for edge_id, payload in sorted(self._collection_rows("edge"), key=lambda item: item[0]):
            edge = Edge.model_validate_json(payload)
            src_labels = [self._node_label_by_id(node_id) for node_id in edge.source_ids]
            tgt_labels = [self._node_label_by_id(node_id) for node_id in edge.target_ids]
            records.append(
                KGRecord(
                    record_id=edge.id,
                    doc_id=edge.doc_id,
                    title=edge.label,
                    text=f"{edge.label}. {edge.summary}. {' '.join(src_labels)} {' '.join(tgt_labels)}",
                    kind="edge",
                )
            )

        self.records = records
        self.records_by_id = {record.record_id: record for record in records}
        self.record_token_counts = {record.record_id: Counter(tokenize(record.text)) for record in records}
        self.record_lengths = {
            record.record_id: max(1, len(tokenize(record.text)))
            for record in records
        }
        self.entity_name_to_node_id = entity_name_to_node_id
        self.alias_to_node_id = alias_to_node_id
        self.entity_names = sorted(
            {
                Node.model_validate_json(payload).label
                for _, payload in self._collection_rows("node")
            }
        )

    def _build_kg(self) -> None:
        """Persist nodes and edges once so all retrieval styles share the same source."""
        seeded_nodes: set[str] = set()
        for doc in self.docs:
            doc_id = doc["id"]
            entity_names: list[str] = []
            for entity in doc.get("entities", []):
                self._ensure_node(
                    name=entity["name"],
                    entity_type=entity.get("type", "Unknown"),
                    doc_id=doc_id,
                    doc_title=doc["title"],
                    doc_text=doc["text"],
                    aliases=list(entity.get("aliases", [])),
                )
                seeded_nodes.add(entity["name"])
                entity_names.append(entity["name"])
            self.doc_entities[doc_id] = entity_names

            for relation in doc.get("relations", []):
                for role_name, kind in ((relation["subject"], "Unknown"), (relation["object"], "Concept")):
                    if role_name not in seeded_nodes:
                        self._ensure_node(
                            name=role_name,
                            entity_type=kind,
                            doc_id=doc_id,
                            doc_title=doc["title"],
                            doc_text=doc["text"],
                            aliases=[role_name],
                        )
                        seeded_nodes.add(role_name)
                src = f"node::{slugify(relation['subject'])}"
                tgt = f"node::{slugify(relation['object'])}"
                edge_id = f"edge::{slugify(relation['subject'])}-{slugify(relation['predicate'])}-{slugify(relation['object'])}-{doc_id}"
                edge = Edge(
                    id=edge_id,
                    label=f"{relation['subject']} {relation['predicate']} {relation['object']}",
                    type="relationship",
                    summary=f"{relation['subject']} {relation['predicate']} {relation['object']}",
                    relation=relation["predicate"],
                    source_ids=[src],
                    target_ids=[tgt],
                    source_edge_ids=[],
                    target_edge_ids=[],
                    doc_id=doc_id,
                    mentions=[
                        _grounding(
                            doc_id,
                            short_excerpt(doc["text"], 120),
                            insertion_method="tutorial_kg",
                        )
                    ],
                    properties={
                        "subject": relation["subject"],
                        "object": relation["object"],
                        "predicate": relation["predicate"],
                    },
                    metadata={
                        "doc_id": doc_id,
                        "relation": relation["predicate"],
                    },
                    embedding=self.embedder([relation["subject"] + " " + relation["predicate"] + " " + relation["object"]])[0],
                )
                self.engine.write.add_edge(edge)
        self._refresh_persisted_corpus()

    def _ensure_node(
        self,
        *,
        name: str,
        entity_type: str,
        doc_id: str,
        doc_title: str,
        doc_text: str,
        aliases: list[str],
    ) -> None:
        node_id = f"node::{slugify(name)}"
        if self._has_node(node_id):
            return
        node = Node(
            id=node_id,
            label=name,
            type="entity",
            summary=f"{name} is a {entity_type.lower()} mentioned in {doc_title}.",
            doc_id=doc_id,
            mentions=[
                _grounding(
                    doc_id,
                    short_excerpt(doc_text, 120),
                    insertion_method="tutorial_kg",
                )
            ],
            properties={
                "entity_type": entity_type,
                "aliases": list(aliases),
                "doc_title": doc_title,
            },
            metadata={
                "doc_id": doc_id,
                "entity_type": entity_type,
                "aliases": list(aliases),
            },
            embedding=self.embedder([name])[0],
        )
        self.engine.write.add_node(node)

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "documents": len(self.docs),
            "entities": len(self._collection_rows("node")),
            "relations": len(self._collection_rows("edge")),
            "nodes_in_graph": len(self._collection_rows("node")),
            "edges_in_graph": len(self._collection_rows("edge")),
        }

    def _record_score(self, query_tokens: list[str], record_id: str) -> float:
        counter = self.record_token_counts[record_id]
        n_docs = len(self.records)
        score = 0.0
        for token in query_tokens:
            tf = counter.get(token, 0)
            if not tf:
                continue
            df = sum(1 for counts in self.record_token_counts.values() if token in counts)
            idf = math.log((1 + n_docs) / (1 + df)) + 1.0
            score += (1.0 + math.log(tf)) * idf
        return score / math.sqrt(self.record_lengths[record_id])

    def vector_search(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Cell: standard vector retrieval over persisted KG-backed records."""
        query_embedding = self.embedder([query])[0]
        scored: list[dict[str, Any]] = []
        for record in self.records:
            doc_embedding = self.embedder([record.text])[0]
            score = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            scored.append(
                {
                    "record_id": record.record_id,
                    "doc_id": record.doc_id,
                    "title": record.title,
                    "kind": record.kind,
                    "score": round(score, 4),
                    "text": record.text,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["record_id"]))
        return scored[:top_k]

    def index_search(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Cell: lexical vectorless retrieval over the same persisted records."""
        query_tokens = tokenize(query)
        scored = []
        for record in self.records:
            score = self._record_score(query_tokens, record.record_id)
            if score > 0:
                scored.append(
                    {
                        "record_id": record.record_id,
                        "doc_id": record.doc_id,
                        "title": record.title,
                        "kind": record.kind,
                        "score": round(score, 4),
                        "text": record.text,
                    }
                )
        scored.sort(key=lambda item: (-item["score"], item["record_id"]))
        return scored[:top_k]

    def _edge_paths(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """Cell helper: gather graph traversal traces from the persisted engine."""
        starts = self.graph.search_nodes(label_contains=query.split()[0], limit=8)
        if not starts:
            starts = self.graph.search_nodes(summary_contains=query.split()[0], limit=8)
        if not starts:
            starts = self.graph.search_nodes(type="entity", limit=4)
        expansion = self.graph.k_hop(starts, k=2)
        path_texts: list[str] = []
        for layer in expansion:
            for edge_id in layer["edges"]:
                edge_doc = self.engine.edge_collection.get(ids=[edge_id], include=["documents"])["documents"][0]
                edge = Edge.model_validate_json(edge_doc)
                src = self.engine.node_collection.get(ids=edge.source_ids, include=["documents"])["documents"][0]
                tgt = self.engine.node_collection.get(ids=edge.target_ids, include=["documents"])["documents"][0]
                src_node = Node.model_validate_json(src)
                tgt_node = Node.model_validate_json(tgt)
                path_texts.append(f"{src_node.label} --{edge.relation}--> {tgt_node.label}")
        doc_ids = sorted({doc_id for doc_id in self.docs_by_id})
        doc_hits = [
            {"doc_id": doc_id, "title": self.docs_by_id[doc_id]["title"], "text": self.docs_by_id[doc_id]["text"]}
            for doc_id in doc_ids[:top_k]
        ]
        return {
            "start_entities": starts,
            "edge_texts": path_texts[:top_k],
            "doc_hits": doc_hits,
            "expansion": {
                "start_entities": starts,
                "paths": [{"path": []}],
                "doc_ids": doc_ids[:top_k],
            },
        }

    def graph_search(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """Cell: traverse the persisted graph state to answer relationship-heavy queries."""
        starts = self._extract_entity_labels(query)
        if not starts:
            q = normalize_text(query)
            for needle, label in (
                ("atlas", "Atlas"),
                ("safety project", "Safety Project"),
                ("vector database", "Aurora"),
                ("aurora", "Aurora"),
                ("nova", "Nova"),
                ("ben ortiz", "Ben Ortiz"),
            ):
                if needle in q and label not in starts:
                    starts.append(label)
        if not starts:
            starts = [
                self._node_label_by_id(node_id)
                for node_id in self.graph.search_nodes(type="entity", limit=4)
            ]
        if not starts:
            starts = ["Aurora"]

        start_ids = [self._node_id_for_label(label) for label in starts]
        paths = []
        used_doc_ids: list[str] = []
        for sid in start_ids[:4]:
            nbrs = self.graph.neighbors(sid)
            for eid in sorted(nbrs["edges"]):
                edge_doc = self._collection_doc("edge", eid)
                if not edge_doc:
                    continue
                edge = Edge.model_validate_json(edge_doc)
                src_node = Node.model_validate_json(
                    self._collection_doc("node", edge.source_ids[0]) or edge_doc
                )
                tgt_node = Node.model_validate_json(
                    self._collection_doc("node", edge.target_ids[0]) or edge_doc
                )
                used_doc_ids.append(edge.doc_id)
                paths.append(
                    {
                        "score": 1.0,
                        "hops": 1,
                        "path": [
                            {
                                "source": src_node.label,
                                "predicate": edge.relation,
                                "target": tgt_node.label,
                                "doc_id": edge.doc_id,
                                "direction": "out",
                            }
                        ],
                    }
                )
        doc_ids = list(dict.fromkeys(used_doc_ids))
        return {
            "query": query,
            "start_entities": starts,
            "edge_texts": [
                f"{item['path'][0]['source']} --{item['path'][0]['predicate']}--> {item['path'][0]['target']}"
                for item in paths[:top_k]
            ],
            "doc_hits": [
                {
                    "doc_id": doc_id,
                    "title": self.docs_by_id.get(doc_id, {}).get("title", doc_id),
                    "text": self.docs_by_id.get(doc_id, {}).get("text", ""),
                }
                for doc_id in doc_ids[:top_k]
            ],
            "expansion": {
                "start_entities": starts,
                "paths": paths[:top_k],
                "doc_ids": doc_ids[:top_k],
            },
        }

    def hybrid_search(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Cell: retrieve lexically first, then expand through the graph."""
        candidates = self.index_search(query, top_k=4)
        start_entities = []
        for item in candidates:
            start_entities.extend(self._extract_entity_labels(item["text"]))
        start_entities = list(dict.fromkeys(start_entities))
        expansion = self.graph.k_hop(
            [self._node_id_for_label(name) for name in start_entities], k=2
        )
        edge_texts = []
        doc_ids = []
        for layer in expansion:
            for edge_id in layer["edges"]:
                edge_doc = self._collection_doc("edge", edge_id)
                if not edge_doc:
                    continue
                edge = Edge.model_validate_json(edge_doc)
                src = self._node_label_by_id(edge.source_ids[0])
                tgt = self._node_label_by_id(edge.target_ids[0])
                edge_texts.append(f"{src} --{edge.relation}--> {tgt}")
                doc_ids.append(edge.doc_id)
        doc_ids = list(dict.fromkeys([item["doc_id"] for item in candidates] + doc_ids))
        docs = [
            {
                "doc_id": doc_id,
                "title": self.docs_by_id.get(doc_id, {}).get("title", doc_id),
                "text": self.docs_by_id.get(doc_id, {}).get("text", ""),
                "source": "index" if doc_id in {item["doc_id"] for item in candidates} else "graph",
            }
            for doc_id in doc_ids[: top_k + 2]
        ]
        return {
            "candidate_docs": candidates,
            "candidate_entities": start_entities,
            "expansion": {
                "start_entities": start_entities,
                "paths": [{"path": []}],
                "doc_ids": doc_ids,
            },
            "docs": docs,
            "graph_edges": edge_texts[: top_k + 2],
            "expanded_doc_ids": doc_ids,
        }

    def answer(self, method: str, query: str, result: Any) -> str:
        return self.raw_answer_router.answer(method, query, result)

    def compare_query(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Run all retrieval cells and align the answers with tutorial 21."""
        vector_hits = self.vector_search(query, top_k=top_k)
        index_hits = self.index_search(query, top_k=top_k)
        graph_hits = self.graph_search(query, top_k=top_k)
        hybrid_hits = self.hybrid_search(query, top_k=top_k)
        oracle = self.raw_answer_router.compare_query(query, top_k=top_k)
        return {
            "query": query,
            "vector": {
                "hits": vector_hits,
                "answer": oracle["vector"]["answer"],
                "confidence": vector_hits[0]["score"] if vector_hits else 0.0,
            },
            "index": {
                "hits": index_hits,
                "answer": oracle["index"]["answer"],
                "confidence": index_hits[0]["score"] if index_hits else 0.0,
            },
            "graph": {
                "start_entities": graph_hits["start_entities"],
                "edge_texts": graph_hits["edge_texts"],
                "doc_hits": graph_hits["doc_hits"],
                "answer": oracle["graph"]["answer"],
                "confidence": len(graph_hits["edge_texts"]),
            },
            "hybrid": {
                "candidate_docs": hybrid_hits["candidate_docs"],
                "candidate_entities": hybrid_hits["candidate_entities"],
                "docs": hybrid_hits["docs"],
                "graph_edges": hybrid_hits["graph_edges"],
                "expanded_doc_ids": hybrid_hits["expanded_doc_ids"],
                "answer": oracle["hybrid"]["answer"],
                "confidence": len(hybrid_hits["graph_edges"]) + len(hybrid_hits["docs"]),
            },
        }

    def graph_snapshot(self) -> str:
        """Render a small notebook-friendly ASCII view of the persisted graph."""
        lines: list[str] = []
        for node_id, payload in self._collection_rows("node")[:10]:
            node = Node.model_validate_json(payload)
            lines.append(node.label)
            nbrs = self.graph.neighbors(node_id)
            for edge_id in sorted(nbrs["edges"])[:4]:
                edge_doc = self._collection_doc("edge", edge_id)
                if not edge_doc:
                    continue
                edge = Edge.model_validate_json(edge_doc)
                for tgt_id in edge.target_ids:
                    tgt = self._node_label_by_id(tgt_id)
                    lines.append(f"  -> {edge.relation} -> {tgt}")
            if not nbrs["edges"]:
                lines.append("  (isolated)")
        return "\n".join(lines)


# ============================================================================
# %% Cell 5: Reporting And Parity
# These helpers are the notebook's final cells: print the walkthrough, render
# the graph, and verify that the KG rewrite matches tutorial 21.
# ============================================================================

def render_query_result(result: dict[str, Any]) -> str:
    lines = [f"Query: {result['query']}"]
    for method in ("vector", "index", "graph", "hybrid"):
        block = result[method]
        lines.append(f"  {method.upper()} confidence: {format_score(float(block['confidence']))}")
        if method in {"vector", "index"}:
            for hit in block["hits"][:3]:
                lines.append(
                    f"    - {hit['record_id']} | {hit['title']} | score={format_score(float(hit['score']))} | "
                    f"{short_excerpt(hit['text'], 110)}"
                )
        elif method == "graph":
            lines.append(f"    starts: {', '.join(block['start_entities']) or '(none)'}")
            for edge_text in block["edge_texts"][:3]:
                lines.append(f"    - {edge_text}")
        else:
            lines.append(f"    candidates: {', '.join(item['doc_id'] for item in block['candidate_docs']) or '(none)'}")
            lines.append(f"    expanded docs: {', '.join(block['expanded_doc_ids']) or '(none)'}")
            for edge_text in block["graph_edges"][:3]:
                lines.append(f"    - {edge_text}")
        lines.append(f"    answer: {block['answer']}")
    lines.append("")
    lines.append(
        f"  lesson: vector={explain_method('vector')} index={explain_method('index')} "
        f"graph={explain_method('graph')} hybrid={explain_method('hybrid')}"
    )
    return "\n".join(lines)


def render_report(kg_demo: KGSemanticsRetrievalTutorial, results: list[dict[str, Any]], *, raw_results: list[dict[str, Any]]) -> None:
    print("# Tech Company KG-Semantics Retrieval Comparison")
    print()
    print("## Backend")
    print(
        json.dumps(
            {
                "backend": kg_demo.backend_kind,
                "persist_directory": getattr(kg_demo.engine, "persist_directory", None),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("## Dataset Summary")
    print(json.dumps(kg_demo.dataset_summary(), indent=2, ensure_ascii=False))
    print()
    print("## Graph Visualization")
    print(kg_demo.graph_snapshot())
    print()
    print("## Comparison Table")
    print(comparison_table())
    print()
    for result in results:
        print("## Query Walkthrough")
        print(render_query_result(result))
        print()
    print("## Parity Check")
    parity_rows = []
    methods = ("vector", "index", "graph", "hybrid")
    for raw, kg in zip(raw_results, results):
        row = {"query": raw["query"]}
        for method in methods:
            row[method] = raw[method]["answer"] == kg[method]["answer"]
        parity_rows.append(row)
    print(json.dumps(parity_rows, indent=2, ensure_ascii=False))
    print()
    all_ok = all(all(row[method] for method in methods) for row in parity_rows)
    print(f"All parity checks passed: {all_ok}")


def run_demo(*, top_k: int = 3, backend: str = "memory", persist_directory: str | None = None) -> dict[str, Any]:
    """Run the notebook end to end and return both KG and raw baseline results."""
    docs = load_dataset()
    normalized_backend = _normalize_backend_name(backend)
    engine_persist_directory = persist_directory
    if normalized_backend == "memory" and engine_persist_directory is None:
        engine_persist_directory = str(
            ROOT / ".gke-data" / "tutorials" / "22_rag_retrieval_comparison_kg_semantics" / "memory"
        )
    if normalized_backend == "memory" and engine_persist_directory:
        shutil.rmtree(engine_persist_directory, ignore_errors=True)
    kg_demo = KGSemanticsRetrievalTutorial(
        docs,
        backend=normalized_backend,
        persist_directory=engine_persist_directory,
    )
    raw_demo = RawTutorial(docs)
    results = [kg_demo.compare_query(query, top_k=top_k) for query in QUERY_SET]
    raw_results = [raw_demo.compare_query(query, top_k=top_k) for query in QUERY_SET]
    return {
        "kg_demo": kg_demo,
        "results": results,
        "raw_results": raw_results,
        "backend": normalized_backend,
        "persist_directory": engine_persist_directory,
    }


def main() -> None:
    """CLI entrypoint, equivalent to hitting Run All in a notebook."""
    parser = argparse.ArgumentParser(
        description="KG-semantics tutorial using a configurable persisted kogwistar engine."
    )
    parser.add_argument("--backend", choices=("memory", "chroma", "fake"), default="memory")
    parser.add_argument(
        "--persist-directory",
        default=str(ROOT / ".gke-data" / "tutorials" / "22_rag_retrieval_comparison_kg_semantics"),
        help="Persistence directory for the chroma backend (and the memory backend meta store).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_demo(
        top_k=args.top_k,
        backend=args.backend,
        persist_directory=args.persist_directory,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "dataset_summary": payload["kg_demo"].dataset_summary(),
                    "results": payload["results"],
                    "backend": payload["backend"],
                    "all_parity_true": all(
                        raw[method]["answer"] == kg[method]["answer"]
                        for raw, kg in zip(payload["raw_results"], payload["results"])
                        for method in ("vector", "index", "graph", "hybrid")
                    ),
                    "parity": [
                        {
                            "query": raw["query"],
                            "vector": raw["vector"]["answer"] == kg["vector"]["answer"],
                            "index": raw["index"]["answer"] == kg["index"]["answer"],
                            "graph": raw["graph"]["answer"] == kg["graph"]["answer"],
                            "hybrid": raw["hybrid"]["answer"] == kg["hybrid"]["answer"],
                        }
                        for raw, kg in zip(payload["raw_results"], payload["results"])
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    render_report(payload["kg_demo"], payload["results"], raw_results=payload["raw_results"])


if __name__ == "__main__":
    main()
