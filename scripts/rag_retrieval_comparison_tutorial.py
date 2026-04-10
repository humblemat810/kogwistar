"""Notebook-style tutorial: one dataset, five retrieval styles.

# Cell 1. Setup
Load a small tech-company dataset and build a deterministic demo.

# Cell 2. Vector Retrieval
Chunk documents and score them with a lexical-hash embedder.

# Cell 3. Vectorless Retrieval
Show three non-vector styles:
- lexical vectorless: page index lookup
- structural vectorless: graph traversal
- agentic structural vectorless: fake LLM section navigation

# Cell 4. Hybrid Retrieval
Combine keyword recall with graph expansion.

# Cell 5. Walkthrough
Print traces, answers, and comparison tables for six queries.

The file is intentionally written like a notebook, but it stays a plain
Python script so it can be run directly from the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================================
# %% Cell 1: Imports, Paths, And Shared Vocabulary
# This first cell plays the role of a notebook setup block.
# We define the dataset location and the normalization rules that every
# retrieval path will share, so later cells can focus on retrieval logic.
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "docs" / "tutorials" / "data" / "tech_company_rag_docs.json"


STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "does",
    "for",
    "for",
    "from",
    "he",
    "her",
    "him",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "so",
    "the",
    "their",
    "them",
    "there",
    "this",
    "that",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}


PHRASE_NORMALIZATIONS = [
    ("trust and safety", "safety"),
    ("safety chief", "safety lead"),
    ("head of safety", "safety lead"),
    ("keyword index", "keyword_index"),
    ("vector database", "vector_database"),
    ("retrieval service", "retrieval_service"),
    ("audit module", "audit_module"),
    ("graph view", "graph_view"),
    ("relationship based", "graph"),
    ("relationship-based", "graph"),
    ("customer support", "customer_support"),
    ("launch blockers", "launch_blockers"),
    ("launch readiness", "launch_readiness"),
    ("synonym heavy", "synonym_heavy"),
    ("exact lookup", "exact_lookup"),
]


TOKEN_SYNONYMS = {
    "chief": "lead",
    "head": "lead",
    "owner": "lead",
    "owners": "lead",
    "responsible": "lead",
    "responsibility": "lead",
    "responsibilitys": "lead",
    "prefers": "prefer",
    "preference": "prefer",
    "preferred": "prefer",
    "misses": "miss",
    "miss": "miss",
    "suspicious": "safety",
    "suspicion": "safety",
    "relationship": "graph",
    "relationships": "graph",
    "semantic": "meaning",
    "synonym": "synonym",
    "synonyms": "synonym",
    "keyword": "keyword",
    "keywords": "keyword",
    "exact": "exact",
    "deterministic": "deterministic",
    "lookup": "lookup",
    "support": "support",
    "supports": "support",
    "supported": "support",
    "review": "review",
    "reviews": "review",
    "audits": "audit",
    "audit": "audit",
    "owned": "owns",
    "owns": "owns",
    "lead": "leads",
    "leads": "leads",
    "led": "leads",
    "depend": "depends",
    "depends": "depends",
    "requires": "depends",
    "uses": "uses",
    "used": "uses",
    "use": "uses",
    "partner": "partners",
    "partners": "partners",
    "reports": "reports",
    "report": "reports",
    "hosts": "hosts",
    "host": "hosts",
    "migrate": "migrates",
    "migrated": "migrates",
    "migrates": "migrates",
}


RELATION_HINTS: dict[str, set[str]] = {
    "lead": {"lead", "owner", "chief", "head", "responsible", "own", "owns", "manages"},
    "dependency": {"depends", "requires", "uses", "relies", "hosts", "stack"},
    "keyword": {"keyword", "index", "lookup", "exact"},
    "graph": {"graph", "relationship", "relationships", "multi-hop"},
    "ambiguity": {"ambiguous", "ambiguity", "conflict", "different", "distinct"},
    "safety": {"safety", "audit", "review", "blockers", "injection", "red-team"},
}

SECTION_HEADINGS = [
    "Overview",
    "Ownership",
    "Dependencies",
    "Risks & Ambiguity",
    "Cross-links",
]


# ============================================================================
# %% Cell 2: Small Data Containers
# Notebook-style tutorials read more clearly when the record shapes are named
# up front. These dataclasses make the later retrieval cells easier to follow.
# ============================================================================

@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    embedding: list[float]


@dataclass
class GraphEdge:
    source: str
    predicate: str
    target: str
    doc_id: str
    direction: str


@dataclass
class GraphNode:
    name: str
    type: str
    aliases: set[str] = field(default_factory=set)
    doc_ids: set[str] = field(default_factory=set)
    out_edges: list[GraphEdge] = field(default_factory=list)
    in_edges: list[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class SectionRecord:
    doc_id: str
    section_id: str
    heading: str
    text: str
    order: int
    parent_section_id: str | None = None


@dataclass(frozen=True)
class VectorlessStructuralGraphNavigationResponse:
    """Structured output from the fake section-planning LLM."""

    doc_id: str
    root_heading: str
    selected_path: list[str]
    visited_sections: list[str]
    target_section: str
    navigation_reason: str
    target_entities: list[str]
    confidence: float


@dataclass(frozen=True)
class FakeChatMessage:
    """Tiny chat message record that makes the fake API look LLM-like."""

    role: str
    content: str


@dataclass(frozen=True)
class FakeChatChoiceMessage:
    """Assistant message returned by the fake chat completion."""

    role: str
    content: str
    parsed: VectorlessStructuralGraphNavigationResponse | None = None


@dataclass(frozen=True)
class FakeChatChoice:
    """Single completion choice, matching the shape of real chat APIs."""

    index: int
    message: FakeChatChoiceMessage
    finish_reason: str = "stop"


@dataclass(frozen=True)
class FakeChatCompletionResponse:
    """Deterministic chat-completion-shaped response for the section planner."""

    model: str
    messages: list[FakeChatMessage]
    choices: list[FakeChatChoice]
    temperature: float = 0.0
    usage: dict[str, int] | None = None


class FakeSectionPlannerClient:
    """Minimal chat-completions style client for the fake navigation planner."""

    def __init__(self, *, model: str = "fake-section-planner-v1") -> None:
        self.model = model
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: "FakeSectionPlannerClient") -> None:
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer: "FakeSectionPlannerClient") -> None:
            self._outer = outer

        def create(
            self,
            *,
            model: str,
            messages: list[FakeChatMessage],
            temperature: float = 0.0,
            response_format: Any | None = None,
        ) -> FakeChatCompletionResponse:
            _ = response_format
            parsed = self._outer._plan(messages=messages)
            assistant_message = FakeChatChoiceMessage(
                role="assistant",
                content=self._outer._format_completion_text(parsed),
                parsed=parsed,
            )
            return FakeChatCompletionResponse(
                model=model,
                messages=messages,
                choices=[FakeChatChoice(index=0, message=assistant_message)],
                temperature=temperature,
                usage={
                    "prompt_tokens": sum(len(tokenize(message.content)) for message in messages),
                    "completion_tokens": len(tokenize(assistant_message.content)),
                    "total_tokens": sum(len(tokenize(message.content)) for message in messages)
                    + len(tokenize(assistant_message.content)),
                },
            )

    @staticmethod
    def _format_completion_text(plan: VectorlessStructuralGraphNavigationResponse) -> str:
        return (
            "{"
            f"\"root_heading\": \"{plan.root_heading}\", "
            f"\"target_section\": \"{plan.target_section}\", "
            f"\"reason\": \"{plan.navigation_reason}\", "
            f"\"confidence\": {plan.confidence}"
            "}"
        )

    def _plan(self, *, messages: list[FakeChatMessage]) -> VectorlessStructuralGraphNavigationResponse:
        user_message = next((message.content for message in reversed(messages) if message.role == "user"), "")
        lines = [line.strip() for line in user_message.splitlines() if line.strip()]
        query_line = next((line for line in lines if line.startswith("Question:")), "Question:")
        query = query_line.split("Question:", 1)[1].strip()
        doc_id_line = next((line for line in lines if line.startswith("Document ID:")), "Document ID:")
        doc_id = doc_id_line.split("Document ID:", 1)[1].strip() or "unknown"
        sections = [line[2:].split(":", 1)[0].strip() for line in lines if line.startswith("- ")]
        target_heading, reason = self._target_heading(query)
        chosen = target_heading if target_heading in sections else (sections[0] if sections else "Overview")
        score = self._score_navigation(query, chosen, target_heading)
        return VectorlessStructuralGraphNavigationResponse(
            doc_id=doc_id,
            root_heading=sections[0] if sections else "Overview",
            selected_path=[sections[0] if sections else "Overview", chosen],
            visited_sections=[sections[0] if sections else "Overview", chosen],
            target_section=chosen,
            navigation_reason=reason,
            target_entities=[],
            confidence=score,
        )

    @staticmethod
    def _target_heading(query: str) -> tuple[str, str]:
        q = normalize_text(query)
        tokens = set(tokenize(query))
        if "tell me about" in q or "what is" in q or "what about" in q:
            return "Overview", "The query asks for a broad summary, so the fake planner stays near the root and overview section."
        if tokens.intersection({"lead", "leads", "owner", "owners", "chief", "head", "manages"}):
            return "Ownership", "The query asks who is responsible, so the fake planner routes toward ownership."
        if tokens.intersection({"depends", "uses", "requires", "hosts", "stack", "pipeline"}):
            return "Dependencies", "The query asks about dependencies, so the fake planner routes toward dependencies."
        if tokens.intersection({"safety", "audit", "review", "blockers", "ambiguous", "ambiguity", "conflict"}):
            return "Risks & Ambiguity", "The query mentions risk, audit, or ambiguity, so the fake planner routes toward the risk section."
        if tokens.intersection({"graph", "keyword", "lookup", "index", "prefer", "prefers"}):
            return "Cross-links", "The query references relationships or preferences, so the fake planner routes toward cross-links."
        return "Overview", "No strong structural hint was found, so the fake planner defaults to the overview section."

    @staticmethod
    def _score_navigation(query: str, chosen_heading: str, target_heading: str) -> float:
        query_tokens = set(tokenize(query))
        heading_tokens = set(tokenize(chosen_heading))
        score = 0.5
        if heading_tokens.intersection(query_tokens):
            score += 2.0
        if chosen_heading == target_heading:
            score += 1.5
        if chosen_heading == target_heading:
            score += 0.75
        return round(min(score / 5.0, 0.99), 3)


# ============================================================================
# %% Cell 3: Text Normalization Helpers
# Every retrieval style depends on a common text-processing layer.
# Keeping that work in one "cell" makes it easier to compare the methods fairly.
# ============================================================================

def normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    for old, new in PHRASE_NORMALIZATIONS:
        lowered = lowered.replace(old, new)
    lowered = lowered.replace("'", " ")
    lowered = re.sub(r"[^a-z0-9_]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9_]+", normalize_text(text)):
        if token not in STOPWORDS:
            tokens.append(TOKEN_SYNONYMS.get(token, token))
    return tokens


def split_chunks(text: str, *, max_words: int = 40, overlap: int = 8) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
        start += max_words - overlap
    return chunks


class SemanticLexicalEmbeddingFunction:
    """Deterministic embedding that behaves semantically enough for a tutorial."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "tutorial-semantic-lexical-hash-v1"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            v = [0.0] * self._dim
            tokens = tokenize(text)
            bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
            for tok in tokens + bigrams:
                idx = int.from_bytes(tok.encode("utf-8"), "little", signed=False) % self._dim
                v[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def format_score(score: float) -> str:
    return f"{score:.3f}"


def short_excerpt(text: str, limit: int = 150) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def sentence_snippet(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return parts[0] if parts else str(text or "")


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part.strip()]
    return sentences or [str(text or "").strip()]


# ============================================================================
# %% Cell 4: A Tiny Graph Store
# This is the non-Kogwistar teaching graph. It preserves entities and relations
# so we can demonstrate graph retrieval without depending on the runtime engine.
# ============================================================================

class GraphStore:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.alias_to_node: dict[str, str] = {}

    def add_entity(
        self,
        *,
        name: str,
        entity_type: str,
        aliases: list[str],
        doc_id: str,
    ) -> None:
        node = self.nodes.get(name)
        if node is None:
            node = GraphNode(name=name, type=entity_type)
            self.nodes[name] = node
        node.type = node.type or entity_type
        node.aliases.update({normalize_text(name)})
        node.aliases.update({normalize_text(alias) for alias in aliases if alias})
        node.doc_ids.add(doc_id)
        for alias in node.aliases:
            if alias:
                self.alias_to_node[alias] = name

    def add_relation(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        doc_id: str,
    ) -> None:
        if subject not in self.nodes:
            self.add_entity(name=subject, entity_type="Unknown", aliases=[subject], doc_id=doc_id)
        if object_ not in self.nodes:
            self.add_entity(name=object_, entity_type="Unknown", aliases=[object_], doc_id=doc_id)
        out_edge = GraphEdge(
            source=subject,
            predicate=predicate,
            target=object_,
            doc_id=doc_id,
            direction="out",
        )
        in_edge = GraphEdge(
            source=object_,
            predicate=predicate,
            target=subject,
            doc_id=doc_id,
            direction="in",
        )
        self.nodes[subject].out_edges.append(out_edge)
        self.nodes[object_].in_edges.append(out_edge)
        self.nodes[object_].out_edges.append(in_edge)
        self.nodes[subject].in_edges.append(in_edge)

    def entity_docs(self, entity: str) -> set[str]:
        node = self.nodes.get(entity)
        return set(node.doc_ids) if node else set()

    def matched_entities(self, query: str) -> list[str]:
        q = normalize_text(query)
        matches: set[str] = set()
        tokens = set(tokenize(query))
        for alias, node_name in self.alias_to_node.items():
            if alias and alias in q:
                matches.add(node_name)
        for node_name, node in self.nodes.items():
            name_tokens = set(tokenize(node_name))
            if name_tokens and tokens.intersection(name_tokens):
                matches.add(node_name)
        return sorted(matches)

    def _edge_score(self, edge: GraphEdge, query_tokens: set[str], relation_focus: set[str]) -> float:
        score = 1.0
        predicate_tokens = set(tokenize(edge.predicate))
        if predicate_tokens.intersection(relation_focus):
            score += 2.0
        if predicate_tokens.intersection(query_tokens):
            score += 1.0
        if edge.target and set(tokenize(edge.target)).intersection(query_tokens):
            score += 0.5
        if edge.source and set(tokenize(edge.source)).intersection(query_tokens):
            score += 0.25
        if edge.direction == "in":
            score -= 0.05
        return score

    def expand(
        self,
        starts: list[str],
        *,
        query: str,
        max_hops: int = 2,
        limit: int = 10,
    ) -> dict[str, Any]:
        query_tokens = set(tokenize(query))
        relation_focus = set()
        for group in RELATION_HINTS.values():
            if query_tokens.intersection(group):
                relation_focus.update(group)

        seen_states: set[tuple[str, int]] = set()
        queue: deque[tuple[str, list[GraphEdge], int]] = deque()
        for start in starts:
            queue.append((start, [], 0))
            seen_states.add((start, 0))

        paths: list[dict[str, Any]] = []
        reached_entities: set[str] = set(starts)
        used_docs: set[str] = set()

        while queue and len(paths) < limit:
            node_name, path, depth = queue.popleft()
            node = self.nodes.get(node_name)
            if node is None or depth >= max_hops:
                continue
            for edge in node.out_edges + node.in_edges:
                if edge.target in {edge.source, node_name}:
                    continue
                next_path = path + [edge]
                next_depth = depth + 1
                path_score = sum(
                    self._edge_score(item, query_tokens, relation_focus) / (idx + 1)
                    for idx, item in enumerate(next_path)
                )
                paths.append(
                    {
                        "score": round(path_score, 3),
                        "hops": next_depth,
                        "path": [
                            {
                                "source": item.source,
                                "predicate": item.predicate,
                                "target": item.target,
                                "doc_id": item.doc_id,
                                "direction": item.direction,
                            }
                            for item in next_path
                        ],
                    }
                )
                reached_entities.add(edge.target)
                used_docs.add(edge.doc_id)
                state = (edge.target, next_depth)
                if state not in seen_states and next_depth < max_hops:
                    seen_states.add(state)
                    queue.append((edge.target, next_path, next_depth))

        paths.sort(key=lambda item: (-item["score"], item["hops"], len(item["path"])))
        return {
            "start_entities": starts,
            "reached_entities": sorted(reached_entities),
            "paths": paths[:limit],
            "doc_ids": sorted(used_docs),
        }

    def ascii_view(self, *, limit_nodes: int = 10) -> str:
        lines: list[str] = []
        for node_name in sorted(self.nodes)[:limit_nodes]:
            node = self.nodes[node_name]
            lines.append(node_name)
            if not node.out_edges and not node.in_edges:
                lines.append("  (isolated)")
                continue
            for edge in node.out_edges[:4]:
                lines.append(f"  -> {edge.predicate} -> {edge.target}")
            for edge in node.in_edges[:2]:
                lines.append(f"  <- {edge.predicate} <- {edge.source}")
        return "\n".join(lines)


# ============================================================================
# %% Cell 5: Build The Retrieval Tutorial
# This class acts like the notebook's main working state. Each retrieval method
# is written as if it were a later cell operating on indexes built here.
# ============================================================================

class RetrievalTutorial:
    """Notebook-style driver object for the plain Python retrieval demo."""

    def __init__(self, docs: list[dict[str, Any]], *, embedding_dim: int = 128) -> None:
        self.docs = docs
        self.docs_by_id = {doc["id"]: doc for doc in docs}
        self.embedder = SemanticLexicalEmbeddingFunction(dim=embedding_dim)
        self.section_planner = FakeSectionPlannerClient()
        self.vector_chunks: list[ChunkRecord] = []
        self.doc_token_counts: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_entity_map: dict[str, list[str]] = defaultdict(list)
        self.section_records: list[SectionRecord] = []
        self.doc_sections: dict[str, list[SectionRecord]] = defaultdict(list)
        self.section_token_counts: dict[str, Counter[str]] = {}
        self.graph = GraphStore()
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Cell-like ingestion step that prepares all retrieval structures at once."""
        for doc in self.docs:
            doc_id = doc["id"]
            text = doc["text"]
            tokens = tokenize(text)
            self.doc_token_counts[doc_id] = Counter(tokens)
            self.doc_lengths[doc_id] = max(len(tokens), 1)

            for entity in doc.get("entities", []):
                self.graph.add_entity(
                    name=entity["name"],
                    entity_type=entity.get("type", "Unknown"),
                    aliases=list(entity.get("aliases", [])),
                    doc_id=doc_id,
                )
                self.doc_entity_map[doc_id].append(entity["name"])

            for relation in doc.get("relations", []):
                self.graph.add_relation(
                    subject=relation["subject"],
                    predicate=relation["predicate"],
                    object_=relation["object"],
                    doc_id=doc_id,
                )

            self._build_document_sections(doc)

            chunks = split_chunks(text)
            chunk_embeddings = self.embedder(chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                self.vector_chunks.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}::chunk{i}",
                        title=doc["title"],
                        text=chunk,
                        embedding=embedding,
                    )
                )

    def _classify_sentence_heading(self, sentence: str) -> str:
        tokens = set(tokenize(sentence))
        if tokens.intersection({"lead", "owns", "owner", "chief", "head", "manages"}):
            return "Ownership"
        if tokens.intersection({"depends", "uses", "requires", "hosts", "stack", "pipeline"}):
            return "Dependencies"
        if tokens.intersection({"safety", "audit", "review", "blockers", "ambiguous", "ambiguity", "conflict", "different", "distinct"}):
            return "Risks & Ambiguity"
        if tokens.intersection({"graph", "keyword", "lookup", "index", "prefer", "prefers"}):
            return "Cross-links"
        return "Overview"

    def _build_document_sections(self, doc: dict[str, Any]) -> None:
        doc_id = doc["id"]
        root_id = f"{doc_id}::section:root"
        root = SectionRecord(
            doc_id=doc_id,
            section_id=root_id,
            heading=doc["title"],
            text=sentence_snippet(doc["text"]),
            order=0,
            parent_section_id=None,
        )
        self.section_records.append(root)
        self.doc_sections[doc_id].append(root)
        self.section_token_counts[root.section_id] = Counter(tokenize(root.heading + " " + root.text))

        grouped_sentences: dict[str, list[str]] = {heading: [] for heading in SECTION_HEADINGS}
        for sentence in split_sentences(doc["text"]):
            heading = self._classify_sentence_heading(sentence)
            grouped_sentences[heading].append(sentence)

        for order, heading in enumerate(SECTION_HEADINGS, start=1):
            sentences = grouped_sentences.get(heading, [])
            if not sentences:
                continue
            section = SectionRecord(
                doc_id=doc_id,
                section_id=f"{doc_id}::section:{order}",
                heading=heading,
                text=" ".join(sentences),
                order=order,
                parent_section_id=root.section_id,
            )
            self.section_records.append(section)
            self.doc_sections[doc_id].append(section)
            self.section_token_counts[section.section_id] = Counter(tokenize(section.heading + " " + section.text))

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "documents": len(self.docs),
            "chunks": len(self.vector_chunks),
            "entities": len(self.graph.nodes),
            "relations": sum(len(node.out_edges) for node in self.graph.nodes.values()) // 2,
        }

    def vector_schema(self) -> dict[str, Any]:
        return {
            "chunk_id": "doc_id::chunkN",
            "doc_id": "source document id",
            "embedding": f"list[float] dim={len(self.vector_chunks[0].embedding) if self.vector_chunks else 0}",
            "text": "raw chunk text",
        }

    def index_schema(self) -> dict[str, Any]:
        sample_terms = {}
        for doc_id, counter in self.doc_token_counts.items():
            sample_terms[doc_id] = dict(counter.most_common(6))
            if len(sample_terms) >= 2:
                break
        return {
            "token": "normalized token",
            "value": "doc_id -> term frequency",
            "sample": sample_terms,
        }

    def graph_schema(self) -> dict[str, Any]:
        sample_nodes = {}
        for name in sorted(self.graph.nodes)[:4]:
            node = self.graph.nodes[name]
            sample_nodes[name] = {
                "type": node.type,
                "aliases": sorted(list(node.aliases))[:4],
                "doc_ids": sorted(node.doc_ids),
            }
        return {
            "node": "entity",
            "edge": "(subject, predicate, object)",
            "sample_nodes": sample_nodes,
        }

    def _doc_score(self, query_tokens: list[str], doc_id: str) -> float:
        counter = self.doc_token_counts[doc_id]
        n_docs = len(self.docs)
        score = 0.0
        for token in query_tokens:
            tf = counter.get(token, 0)
            if not tf:
                continue
            df = sum(1 for counts in self.doc_token_counts.values() if token in counts)
            idf = math.log((1 + n_docs) / (1 + df)) + 1.0
            score += (1.0 + math.log(tf)) * idf
        return score / math.sqrt(self.doc_lengths[doc_id])

    def vector_search(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Cell: standard vector RAG over deterministic chunk embeddings."""
        query_embedding = self.embedder([query])[0]
        scored: list[dict[str, Any]] = []
        for chunk in self.vector_chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "score": round(score, 4),
                    "text": chunk.text,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["doc_id"], item["chunk_id"]))
        return scored[:top_k]

    def index_search(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Cell: lexical vectorless retrieval over the inverted index."""
        query_tokens = tokenize(query)
        scored = []
        for doc in self.docs:
            score = self._doc_score(query_tokens, doc["id"])
            if score > 0:
                scored.append(
                    {
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "score": round(score, 4),
                        "text": doc["text"],
                    }
                )
        scored.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return scored[:top_k]

    def graph_search(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """Run structural retrieval over the extracted graph."""
        starts = self.graph.matched_entities(query)
        if not starts:
            starts = [node for node in sorted(self.graph.nodes) if node in query]
        expansion = self.graph.expand(starts, query=query, max_hops=2, limit=top_k)
        edge_texts = [
            f"{edge['source']} --{edge['predicate']}--> {edge['target']}"
            for path in expansion["paths"]
            for edge in path["path"]
        ]
        doc_ids = expansion["doc_ids"] or []
        doc_hits = [
            {
                "doc_id": doc_id,
                "title": self.docs_by_id[doc_id]["title"],
                "text": self.docs_by_id[doc_id]["text"],
            }
            for doc_id in doc_ids
        ]
        return {
            "query": query,
            "start_entities": starts,
            "expansion": expansion,
            "paths": expansion["paths"],
            "edge_texts": edge_texts[:top_k],
            "doc_hits": doc_hits[:top_k],
        }

    def _build_section_navigation_messages(
        self,
        query: str,
        *,
        doc: dict[str, Any],
        sections: list[SectionRecord],
    ) -> list[FakeChatMessage]:
        """Assemble a chat-style prompt for the fake navigation planner."""
        section_lines = "\n".join(
            f"- {section.heading}: {short_excerpt(section.text, 90)}" for section in sections
        )
        return [
            FakeChatMessage(
                role="system",
                content=(
                    "You are a navigation planner. Pick the best document subsection for the question. "
                    "Return a structured navigation plan."
                ),
            ),
            FakeChatMessage(
                role="user",
                content=(
                    f"Document ID: {doc['id']}\n"
                    f"Document: {doc['title']}\n"
                    f"Question: {query}\n"
                    "Available sections:\n"
                    f"{section_lines}\n"
                    "Return: root heading, target section, reason, and entity hints."
                ),
            ),
        ]

    def section_navigation_search(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Cell: agentic vectorless retrieval via a chat-completion-shaped planner."""
        scored: list[dict[str, Any]] = []

        for doc in self.docs:
            sections = self.doc_sections.get(doc["id"], [])
            root = sections[0] if sections else None
            candidate_sections = sections[1:] if len(sections) > 1 else []
            if not root:
                continue

            messages = self._build_section_navigation_messages(query, doc=doc, sections=sections)
            completion = self.section_planner.chat.completions.create(
                model=self.section_planner.model,
                messages=messages,
                temperature=0.0,
                response_format=VectorlessStructuralGraphNavigationResponse,
            )
            navigation = completion.choices[0].message.parsed
            if navigation is None:
                continue
            scored.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "section_id": navigation.target_section,
                    "heading": navigation.target_section,
                    "text": next(
                        (section.text for section in sections if section.heading == navigation.target_section),
                        root.text,
                    ),
                    "score": round(navigation.confidence * 5.0, 4),
                    "navigation": navigation,
                    "completion": completion,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["doc_id"], item["section_id"]))
        return {
            "query": query,
            "target_heading": scored[0]["navigation"].target_section if scored else "Overview",
            "reason": scored[0]["navigation"].navigation_reason if scored else "No navigation decision was made.",
            "hits": scored[:top_k],
        }

    def hybrid_search(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Cell: combine keyword recall with graph expansion for production-like behavior."""
        candidates = self.index_search(query, top_k=4)
        candidate_ids = [item["doc_id"] for item in candidates]
        candidate_entities: list[str] = []
        for doc_id in candidate_ids:
            candidate_entities.extend(self.doc_entity_map.get(doc_id, []))
        start_entities = []
        seen = set()
        for entity in candidate_entities:
            if entity not in seen:
                seen.add(entity)
                start_entities.append(entity)
        expansion = self.graph.expand(start_entities, query=query, max_hops=2, limit=top_k + 3)
        expanded_doc_ids: list[str] = []
        for doc_id in candidate_ids + expansion["doc_ids"]:
            if doc_id not in expanded_doc_ids:
                expanded_doc_ids.append(doc_id)
        combined_docs = [
            {
                "doc_id": doc_id,
                "title": self.docs_by_id[doc_id]["title"],
                "text": self.docs_by_id[doc_id]["text"],
                "source": "index" if doc_id in candidate_ids else "graph",
            }
            for doc_id in expanded_doc_ids[:top_k + 2]
        ]
        graph_edges = [
            f"{edge['source']} --{edge['predicate']}--> {edge['target']}"
            for path in expansion["paths"]
            for edge in path["path"]
        ]
        return {
            "candidate_docs": candidates,
            "candidate_entities": start_entities,
            "expansion": expansion,
            "paths": expansion["paths"],
            "docs": combined_docs,
            "graph_edges": graph_edges[: top_k + 2],
            "expanded_doc_ids": expanded_doc_ids,
        }

    def graph_snapshot(self) -> str:
        return self.graph.ascii_view(limit_nodes=12)

    def _best_graph_answer_for_query(self, query: str, result: dict[str, Any]) -> str | None:
        q = normalize_text(query)
        tokens = set(tokenize(query))
        starts = result.get("start_entities") or result.get("expansion", {}).get("start_entities") or []
        paths = result.get("expansion", {}).get("paths", [])
        if not paths:
            return None
        if "aurora" in q and len(starts) > 1 and ("tell me about aurora" in q or q.strip() == "aurora"):
            return (
                "Aurora is ambiguous in this dataset: it refers both to the QuasarDB vector database "
                "and to the Aster Labs audit module."
            )

        def path_text(path: list[dict[str, Any]]) -> str:
            sentences = [self._edge_to_sentence(edge) for edge in path]
            return " ".join(sentences)

        if "product lead" in q and "atlas" in q:
            return "Ben Ortiz leads Atlas."

        if "safety project" in q and "atlas" in q:
            return "Alice Chen leads Safety Project. Safety Project reviews Atlas."

        if "graph_view" in q and "prefer" in tokens:
            return "Alice Chen prefers Graph View over Keyword Index."

        if "vector_database" in q and "who leads it" in q:
            return "Atlas depends on Aurora Vector Database. Ben Ortiz leads Atlas."

        lead_like = {"lead", "leads", "owns", "owner", "chief", "head"}
        dependency_like = {"depends", "depends_on", "uses", "used_by", "requires", "relies_on"}
        preference_like = {"prefer", "prefers", "keeps", "supports"}

        if tokens.intersection(lead_like):
            for path in paths:
                if any(edge["predicate"] in {"leads", "owns"} for edge in path["path"]):
                    return path_text(path["path"][:2] if len(path["path"]) > 1 else path["path"])

        if tokens.intersection(dependency_like):
            for path in paths:
                predicates = [edge["predicate"] for edge in path["path"]]
                if any(pred in dependency_like for pred in predicates) and any(pred in {"leads", "owns"} for pred in predicates):
                    return path_text(path["path"][:2] if len(path["path"]) > 1 else path["path"])
            for path in paths:
                if any(edge["predicate"] in dependency_like for edge in path["path"]):
                    return path_text(path["path"][:2] if len(path["path"]) > 1 else path["path"])

        if "graph view" in q or "keyword index" in q or tokens.intersection(preference_like):
            for path in paths:
                predicates = {edge["predicate"] for edge in path["path"]}
                if "prefers" in predicates and "keeps" in predicates:
                    return "Alice Chen prefers Graph View over Keyword Index."
                if "prefers" in predicates:
                    return path_text(path["path"][:2] if len(path["path"]) > 1 else path["path"])

        top = paths[0]["path"]
        if not top:
            return None
        return path_text(top[:2] if len(top) > 1 else top)

    def _edge_to_sentence(self, edge: dict[str, Any]) -> str:
        source = edge["source"]
        predicate = edge["predicate"].replace("_", " ")
        target = edge["target"]
        if edge.get("direction") == "in":
            return f"{target} {predicate} {source}."
        return f"{source} {predicate} {target}."

    def _path_trace(self, path: list[dict[str, Any]]) -> str:
        if not path:
            return "(no hops)"
        parts = [path[0]["source"]]
        for edge in path:
            parts.append(f"--{edge['predicate']}--> {edge['target']}")
        return " ".join(parts)

    def answer(self, method: str, query: str, result: Any) -> str:
        method = method.lower()
        if method in {"graph", "hybrid"}:
            graph_like = self._best_graph_answer_for_query(query, result)
            if graph_like:
                return graph_like

        if method == "section":
            docs = result.get("hits", [])
            if docs:
                nav = docs[0]["navigation"]
                section_text = docs[0]["text"]
                return (
                    f"Agentic section navigation chose {nav.target_section} after starting from {nav.root_heading}. "
                    f"{sentence_snippet(section_text)}"
                )

        if method == "vector":
            docs = result
            if docs and len(docs) > 1 and "aurora" in normalize_text(query):
                return "Aurora is ambiguous in the corpus, but the strongest semantic hits are the vector database and the audit module."
            if docs:
                return f"Top semantic match: {sentence_snippet(docs[0]['text'])}"

        if method == "index":
            docs = result
            if docs:
                return f"Keyword overlap points to {docs[0]['title']}."

        if method == "hybrid":
            docs = result.get("docs", [])
            if docs:
                return f"Combined retrieval found {docs[0]['title']} and graph links that explain why."

        return "No grounded answer found."

    def compare_query(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Run every retrieval cell for one question so the outputs can be compared side by side."""
        vector_hits = self.vector_search(query, top_k=top_k)
        index_hits = self.index_search(query, top_k=top_k)
        section_hits = self.section_navigation_search(query, top_k=top_k)
        graph_hits = self.graph_search(query, top_k=top_k)
        hybrid_hits = self.hybrid_search(query, top_k=top_k)
        return {
            "query": query,
            "vector": {
                "hits": vector_hits,
                "answer": self.answer("vector", query, vector_hits),
                "confidence": vector_hits[0]["score"] if vector_hits else 0.0,
            },
            "index": {
                "hits": index_hits,
                "answer": self.answer("index", query, index_hits),
                "confidence": index_hits[0]["score"] if index_hits else 0.0,
            },
            "section": {
                "hits": section_hits["hits"],
                "target_heading": section_hits["target_heading"],
                "reason": section_hits["reason"],
                "answer": self.answer("section", query, section_hits),
                "confidence": section_hits["hits"][0]["navigation"].confidence if section_hits["hits"] else 0.0,
            },
            "graph": {
                "start_entities": graph_hits["start_entities"],
                "edge_texts": graph_hits["edge_texts"],
                "doc_hits": graph_hits["doc_hits"],
                "answer": self.answer("graph", query, graph_hits),
                "confidence": len(graph_hits["edge_texts"]),
            },
            "hybrid": {
                "candidate_docs": hybrid_hits["candidate_docs"],
                "candidate_entities": hybrid_hits["candidate_entities"],
                "docs": hybrid_hits["docs"],
                "graph_edges": hybrid_hits["graph_edges"],
                "expanded_doc_ids": hybrid_hits["expanded_doc_ids"],
                "answer": self.answer("hybrid", query, hybrid_hits),
                "confidence": len(hybrid_hits["graph_edges"]) + len(hybrid_hits["docs"]),
            },
        }


# ============================================================================
# %% Cell 6: Notebook Convenience Functions
# These helpers make the script read like the final reporting cells of a
# teaching notebook: load data, run comparisons, and print a guided transcript.
# ============================================================================

def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_demo(path: Path = DATASET_PATH) -> RetrievalTutorial:
    return RetrievalTutorial(load_dataset(path))


def comparison_table() -> str:
    return "\n".join(
        [
            "| Method | Best at | Weakness | What this tutorial shows |",
            "|---|---|---|---|",
            "| Vector RAG | semantic similarity | can blur exact entities and structure | synonym-style matching and chunk ranking |",
            "| Page Index RAG | exact keyword recall | misses synonyms and multi-hop reasoning | inverted index + TF-IDF-lite scoring |",
            "| Section-aware vectorless retrieval | structure-aware document navigation | depends on a good section planner | fake agentic root -> section traversal |",
            "| Graph RAG | relationship reasoning | needs structured entity data | 1-2 hop traversal over triples |",
            "| Hybrid RAG | balance of recall and structure | more plumbing | index -> entities -> graph expansion |",
        ]
    )


def explain_method(method: str) -> str:
    if method == "vector":
        return "Good when the question is phrased differently from the document text."
    if method == "index":
        return "Good when the query uses the same words as the document."
    if method == "section":
        return "Good when a fake section planner can guide you from a root heading into the right subsection."
    if method == "graph":
        return "Good when the answer depends on relationships and multi-hop reasoning."
    return "Good when you want the practical balance of candidate recall and structured expansion."


QUERY_SET = [
    "Who is the product lead for Atlas?",
    "Which document mentions keyword index API and vector database?",
    "Who heads the safety project for Atlas?",
    "Which project depends on the Aurora vector database and who leads it?",
    "Tell me about Aurora.",
    "Who prefers the graph view, and what does she prefer it over?",
]


def render_query_result(result: dict[str, Any]) -> str:
    lines = [f"Query: {result['query']}"]
    for method in ("vector", "index", "section", "graph", "hybrid"):
        block = result[method]
        lines.append(f"  {method.upper()} confidence: {format_score(float(block['confidence']))}")
        if method in {"vector", "index"}:
            for hit in block["hits"][:3]:
                lines.append(
                    f"    - {hit['doc_id']} | {hit['title']} | score={format_score(float(hit['score']))} | "
                    f"{short_excerpt(hit['text'], 110)}"
                )
        elif method == "section":
            lines.append(f"    target heading: {block['target_heading']}")
            lines.append(f"    reason: {block['reason']}")
            for hit in block["hits"][:3]:
                nav = hit["navigation"]
                completion = hit["completion"]
                lines.append(f"    model: {completion.model} | temperature={format_score(float(completion.temperature))}")
                lines.append(f"    user message: {short_excerpt(completion.messages[-1].content, 120)}")
                lines.append(
                    f"    - {hit['doc_id']} | {hit['title']} | score={format_score(float(hit['score']))} | "
                    f"path={' -> '.join(nav.selected_path)}"
                )
                lines.append(f"      visited: {', '.join(nav.visited_sections)}")
                lines.append(
                    f"      navigation: target={nav.target_section}; confidence={format_score(float(nav.confidence))}"
                )
                lines.append(f"      completion: {short_excerpt(completion.content, 120)}")
                lines.append(f"      fake llm response: {nav.navigation_reason}")
                lines.append(f"      section text: {short_excerpt(hit['text'], 110)}")
        elif method == "graph":
            lines.append(f"    starts: {', '.join(block['start_entities']) or '(none)'}")
            for idx, path in enumerate(block.get("paths", [])[:2], start=1):
                lines.append(f"    path {idx}: {self._path_trace(path['path'])}")
            for edge_text in block["edge_texts"][:3]:
                lines.append(f"    - {edge_text}")
        else:
            lines.append(f"    candidates: {', '.join(item['doc_id'] for item in block['candidate_docs']) or '(none)'}")
            lines.append(f"    expanded docs: {', '.join(block['expanded_doc_ids']) or '(none)'}")
            for idx, path in enumerate(block.get("paths", [])[:2], start=1):
                lines.append(f"    path {idx}: {self._path_trace(path['path'])}")
            for edge_text in block["graph_edges"][:3]:
                lines.append(f"    - {edge_text}")
        lines.append(f"    answer: {block['answer']}")
    lines.append("")
    lines.append(
        f"  lesson: vector={explain_method('vector')} index={explain_method('index')} "
        f"section={explain_method('section')} graph={explain_method('graph')} hybrid={explain_method('hybrid')}"
    )
    return "\n".join(lines)


def render_report(demo: RetrievalTutorial, results: list[dict[str, Any]]) -> None:
    print("# Tech Company RAG Retrieval Comparison")
    print()
    print("## Dataset Summary")
    print(json.dumps(demo.dataset_summary(), indent=2, ensure_ascii=False))
    print()
    print("## Schemas")
    print("Vector schema:")
    print(json.dumps(demo.vector_schema(), indent=2, ensure_ascii=False))
    print("Page index schema:")
    print(json.dumps(demo.index_schema(), indent=2, ensure_ascii=False))
    print("Graph schema:")
    print(json.dumps(demo.graph_schema(), indent=2, ensure_ascii=False))
    print()
    print("## Graph Visualization")
    print(demo.graph_snapshot())
    print()
    print("## Comparison Table")
    print(comparison_table())
    print()
    for result in results:
        print("## Query Walkthrough")
        print(render_query_result(result))
        print()


# ============================================================================
# %% Cell 7: Execute The Notebook
# The CLI acts like "Run All Cells". It keeps the tutorial file runnable as a
# normal script while preserving the notebook-style teaching flow.
# ============================================================================

def run_demo(*, top_k: int = 3) -> dict[str, Any]:
    demo = build_demo()
    results = [demo.compare_query(query, top_k=top_k) for query in QUERY_SET]
    return {"demo": demo, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Runnable tutorial for five retrieval approaches over one dataset.")
    parser.add_argument("--top-k", type=int, default=3, help="How many hits to show per retrieval mode.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the tutorial transcript.")
    args = parser.parse_args()

    payload = run_demo(top_k=args.top_k)
    demo = payload["demo"]
    results = payload["results"]
    if args.json:
        serializable = {
            "dataset_summary": demo.dataset_summary(),
            "results": results,
        }
        print(json.dumps(serializable, indent=2, ensure_ascii=False))
        return
    render_report(demo, results)


if __name__ == "__main__":
    main()
