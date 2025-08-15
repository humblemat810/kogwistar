from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from joblib import Memory
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from .engine import GraphKnowledgeEngine
from .models import (
    Document,
    Edge,
    ReferenceSession,
)
from .models import Node
# --- relation name constants (optional but handy) ---
REL_SUMMARIZES = "summarizes"   # parent -> child
REL_DETAILS    = "details"      # child  -> parent

REL_PRECEDES   = "precedes"     # left sib -> right sib
REL_AFTER      = "after"        # right sib -> left sib

REL_SUMMARIZES_DOCUMENT   = "summarizes_document"     # final_summary -> document_node
REL_DOCUMENT_DETAILED_BY  = "document_detailed_by"    # document_node -> final_summary
# -----------------------------
# Joblib cache (disk-persistent)
# -----------------------------
# Default under the engine's persist directory for easy cleanup/backups
def _default_cache_dir(persist_dir: Optional[str]) -> str:
    base = persist_dir or "./chroma_db"
    return f"{base.rstrip('/')}/.ingester_cache"

# -----------------------------
# LLM Schemas
# -----------------------------

class MiniChunk(BaseModel):
    """Result of 'summarize into smaller chunks' for a single input span."""
    title: str = Field(..., description="Short title for this micro-chunk.")
    summary: str = Field(..., description="Concise summary of this micro-chunk.")
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)

class SummarizeResponse(BaseModel):
    micro_chunks: List[MiniChunk] = Field(
        ..., description="Ordered micro-chunks covering the input span without large overlap."
    )

class GroupItem(BaseModel):
    """Grouping output referencing positions in the provided input list."""
    title: str = Field(..., description="Group title.")
    summary: str = Field(..., description="Higher-level summary of the group.")
    member_indices: List[int] = Field(..., description="Indices into the provided chunk list, in order.")

class GroupResponse(BaseModel):
    groups: List[GroupItem]

# -----------------------------
# Helper datatypes
# -----------------------------

class Span(BaseModel):
    """Logical span within a document. Used to preserve traceability."""
    start_page: int
    end_page: int
    start_char: int
    end_char: int

class LeafChunk(BaseModel):
    """Initial (leaf) chunk from the raw document."""
    id: str
    text: str
    span: Span

class SummaryChunk(BaseModel):
    """A summarized chunk produced by the LLM."""
    id: str
    title: str
    summary: str
    span: Span
    level: int  # 0 for micro from leaf, 1 for first grouping, etc.

# -----------------------------
# Core side-car ingestor
# -----------------------------

@dataclass
class BaseDocumentGraphIngestor:
    engine: GraphKnowledgeEngine
    llm: BaseChatModel
    cache_dir: Optional[str] = None

    def __post_init__(self):
        self.use_uuid = False
        self.memory = Memory(self.cache_dir or _default_cache_dir(getattr(self.engine, "persist_directory", None)), verbose=0)
        # prepare structured-output chains once to make cache keys stable
        self._summarize_chain = self.llm.with_structured_output(SummarizeResponse, include_raw=False)
        self._group_chain = self.llm.with_structured_output(GroupResponse, include_raw=False)

        # Wrap LLM calls with joblib to cache by pure-string inputs
        self._cached_summarize = self.memory.cache(self._summarize_call, ignore=["self"])
        self._cached_group = self.memory.cache(self._group_call, ignore=["self"])

    # ---------- public entrypoints ----------

    def ingest_document(
        self,
        *,
        document: Document,
        split_max_chars: int = 2500,
        group_size: int = 5,
        max_levels: int = 6,
        force_concat_after_levels: int = 3,
    ) -> dict:
        """
        Full side-car pipeline:
        1) split doc -> leaf chunks
        2) summarize each leaf into micro-chunks
        3) group micro-chunks into larger chunks
        4) persist nodes/edges for summaries and adjacency (bidirectional)
        5) iterate 2-3 with the newly created larger chunks until 1 node or max_levels
        6) if still >1 after `force_concat_after_levels`, force-concatenate to final node
        """
        # Ensure the raw document row exists (this does not trigger any LLM extraction)
        self.engine.add_document(document)

        # Step 1: split
        leaves = self._split_into_leaves(document.content or "", split_max_chars, document.id)

        # Persist leaf nodes (type="page" or "leaf") so relationships have concrete endpoints
        leaf_nodes = self._persist_leaf_nodes(document.id, leaves)

        # Build "next_to" adjacency among leaves (bidirectional)
        self._persist_adjacency(document.id, [n.id for n in leaf_nodes])

        # Current layer = leaves -> summarize into micro-chunks (level 0)
        current_layer: List[SummaryChunk] = self._summarize_layer(document.id, leaves, level=0)

        # Persist summarization edges (leaf <-> micro)
        self._persist_layer(document.id, parents=leaf_nodes, children=current_layer)

        # Iterate: group up layer by layer
        level = 1
        layers = []
        while len(current_layer) > 1 and level <= max_levels:
            layers.append(current_layer)
            # Optional early force after N levels if not converging
            if level >= force_concat_after_levels and len(current_layer) > 4 and len(current_layer) / max(1, len(leaf_nodes)) > 0.15:
                final = self._force_summarize(document.id, current_layer, level = level)
                self._persist_layer(document.id, parents=[self._as_node(document.id, c) for c in current_layer], children=[final])
                current_layer = [final]
                break

            # group step produces higher-level chunks
            next_layer = self._group_layer(document.id, current_layer, level=level)

            # persist edges between current_layer (parents) and next_layer (children)
            self._persist_layer(document.id, parents=[self._as_node(document.id, c) for c in current_layer], children=next_layer)
            current_layer = next_layer
            level += 1

        # If still more than 1, finalize by concatenation
        if len(current_layer) > 1:
            final = self._force_summarize(document.id, current_layer, level = level)
            self._persist_layer(document.id, parents=[self._as_node(document.id, c) for c in current_layer], children=[final])
            current_layer = [final]
            
        final = current_layer[0]
        final_node_id = self._ensure_node(document.id, final)  # ensures final chunk node
        docnode_id = self._ensure_document_node(document.id, title=document.metadata.get("title") if document.metadata else None, leaves = leaves)

        # wire asymmetric, document-level relationship
        self._bi_edge(
            document.id,
            src=final_node_id if isinstance(final_node_id, str) else self._as_node(document.id, final).id,
            tgt=docnode_id,
            relation=REL_SUMMARIZES_DOCUMENT,
            reverse_relation=REL_DOCUMENT_DETAILED_BY,
        )
        return {
            "document_id": document.id,
            "leaf_count": len(leaves),
            "levels": level,
            "final_node_id": self._as_node(document.id, current_layer[0]).id if current_layer else None,
        }
    def _ensure_document_node(self, doc_id: str, *, title: str | None = None, leaves) -> str:
        node_id = f"docnode:{doc_id}"
        if not self.engine._exists_node(node_id):
            from graph_knowledge_engine.models import Node, ReferenceSession
            embeddings = self.engine.document_collection.get(doc_id, include = ['embeddings'])['embeddings'][0]
            n = Node(
                id=node_id,
                label=title or f"Document {doc_id}",
                type="document",
                summary="Represents the whole source document.",
                references=[ReferenceSession(
                    collection_page_url=f"document_collection/{doc_id}",
                    document_page_url=f"document/{doc_id}",
                    start_page=1, end_page=len(leaves), start_char=0, end_char=len(leaves[-1].text), snippet=None,
                    doc_id = doc_id
                )],
                doc_id=doc_id,
                properties={"kind": "document_root"},
                embedding = embeddings
            )
            self.engine.add_node(n, doc_id=doc_id)
        return node_id
    # ---------- splitting ----------

    def _split_into_leaves(self, text: str, max_chars: int, doc_id) -> List[LeafChunk]:
        if "\f" in text:  # respect page breaks if present
            raw_pages = [p for p in (seg.strip() for seg in text.split("\f")) if p]
            leaves: List[LeafChunk] = []
            pos_char = 0
            for i, page in enumerate(raw_pages, start=1):
                pid = f"leaf:{uuid.uuid4() if self.use_uuid else i}"
                span = Span(start_page=i, end_page=i, start_char=0, end_char=len(page))
                leaves.append(LeafChunk(id=pid, text=page, span=span))
                pos_char += len(page) + 1
            return leaves

        # greedily pack paragraphs into ~max_chars chunks
        paras = [p for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
        leaves: List[LeafChunk] = []
        buf: List[str] = []
        cur_len = 0
        start_char_global = 0
        page_counter = 1  # if no page info, keep page=1 for all
        i = 0
        for p in paras:
            p_len = len(p)
            if buf and cur_len + 2 + p_len > max_chars:
                i += 1
                chunk_text = "\n\n".join(buf)
                cid = f"leaf:{uuid.uuid4() if self.use_uuid else i}"
                span = Span(start_page=page_counter, end_page=page_counter, start_char=0, end_char=len(chunk_text))
                leaves.append(LeafChunk(id=cid, text=chunk_text, span=span))
                start_char_global += len(chunk_text) + 2
                buf, cur_len = [p], p_len
            else:
                if buf:
                    cur_len += 2 + p_len
                else:
                    cur_len = p_len
                buf.append(p)
        if buf:
            i +=1
            chunk_text = "\n\n".join(buf)
            cid = f"leaf:{uuid.uuid4() if self.use_uuid else i}|doc:{doc_id}"
            span = Span(start_page=page_counter, end_page=page_counter, start_char=0, end_char=len(chunk_text))
            leaves.append(LeafChunk(id=cid, text=chunk_text, span=span))

        return leaves or [LeafChunk(id=f"leaf:{uuid.uuid4() if self.use_uuid else i+1}|doc:{doc_id}", text=text.strip(), span=Span(start_page=1, end_page=1, start_char=0, end_char=len(text.strip())))]

    # ---------- LLM calls (cached) ----------

    def _summarize_call(self, prompt_key: str, content: str): # -> SummarizeResponse:
        # single, stable prompt
        system = (
            "You are a precise technical summarizer. "
            "Break the given text into 2–8 coherent micro-chunks, preserving order. "
            "Each micro-chunk must include a short title, a concise summary, and span indices "
            "(start_page, end_page, start_char, end_char) local to the provided text if pages exist; "
            "otherwise use page=1."
        )
        messages = [
            ("system", system),
            ("human", f"TEXT:\n{content}"),
        ]
        chain = self._summarize_chain  # structured to SummarizeResponse
        to_return = chain.invoke(messages)
        return to_return

    def _group_call(self, prompt_key: str, chunk_titles_and_summaries: str, max_groups: int): # -> GroupResponse:
        system = (
            "You are a careful text organizer. "
            f"Group the provided items into ≤{max_groups} higher-level groups. "
            "Preserve order as much as possible. Each group must contain contiguous items. "
            "Return titles, summaries, and member indices (0-based)."
        )
        messages = [
            ("system", system),
            ("human", f"ITEMS:\n{chunk_titles_and_summaries}"),
        ]
        chain = self._group_chain  # structured to GroupResponse
        return chain.invoke(messages)
    
    # ---------- summarize/group layers ----------
    def _force_summarize(self, doc_id: str, chunks: list[SummaryChunk], *, level: int) -> SummaryChunk:
        combined_text = "\n".join(f"{i+1}. {c.summary}" for i, c in enumerate(chunks))
        key = f"force_sum:v1|doc:{doc_id}|level:{level}"
        res = self._cached_summarize(key, combined_text)
        summary_resp: SummarizeResponse = SummarizeResponse.model_validate(res)
        # expect exactly one micro_chunk back
        mc = summary_resp.micro_chunks[0]
        final_span = Span(
            start_page=min(c.span.start_page for c in chunks),
            end_page=max(c.span.end_page for c in chunks),
            start_char=0,
            end_char=max(c.span.end_char for c in chunks)
        )
        final_chunk = SummaryChunk(
            id=f"final:{uuid.uuid4()}",
            title=mc.title or "Final Summary",
            summary=mc.summary,
            span=final_span,
            level=level
        )
        self._ensure_node(doc_id, final_chunk)
        for c in chunks:
            self._bi_edge(doc_id, src=final_chunk.id, tgt=c.id,
                        relation=REL_SUMMARIZES, reverse_relation=REL_DETAILS)
        return final_chunk
    def _summarize_layer(self, doc_id: str, leaves: Sequence[LeafChunk], *, level: int) -> List[SummaryChunk]:
        out: List[SummaryChunk] = []
        counts_per_leaf: List[int] = []

        # 1) Summarize each leaf into micro-chunks (collect SummaryChunk objects)
        for leaf in leaves:
            key = f"sum:v1|doc:{doc_id}|leaf:{leaf.id}|level:{level}"
            raw = self._cached_summarize(key, leaf.text)
            res: SummarizeResponse = SummarizeResponse.model_validate(raw)

            local_cnt = 0
            for i, m in enumerate(res.micro_chunks):
                # normalize spans to be within 1 page if no page break info is meaningful
                span = Span(
                    start_page=max(1, m.start_page or leaf.span.start_page),
                    end_page=max(1, m.end_page or leaf.span.end_page),
                    start_char=max(0, m.start_char),
                    end_char=max(m.start_char, m.end_char),
                )
                out.append(
                    SummaryChunk(
                        id=(f"sum:{uuid.uuid4()}" if self.use_uuid else f"sum|doc:{doc_id}|level:{level}|c{i}"),
                        title=(m.title or "").strip()[:120] or "Untitled",
                        summary=(m.summary or "").strip(),
                        span=span,
                        level=level,
                    )
                )
                local_cnt += 1
            counts_per_leaf.append(local_cnt)

        # 2) Persist micro-chunk nodes, then sibling adjacency per leaf
        offset = 0
        for cnt in counts_per_leaf:
            if cnt == 0:
                continue
            subset = out[offset:offset + cnt]          # the micro-chunks from this leaf
            self._persist_micro_chunks_as_nodes(doc_id, subset)   # <-- persist nodes first
            micros_ids = [c.id for c in subset]
            self._persist_adjacency(doc_id, micros_ids)           # then wire adjacency (precedes/after)
            offset += cnt

        return out

    def _group_layer(self, doc_id: str, current: Sequence[SummaryChunk], *, level: int) -> List[SummaryChunk]:
        """Group consecutive chunks into higher-level chunks, using LLM grouping with caching."""
        # Prepare the grouping text
        items_text = "\n".join(f"[{i}] {c.title}: {c.summary}" for i, c in enumerate(current))
        max_groups = max(1, (len(current) + 1) // 2)  # heuristic: halve the count
        key = f"group:v1|doc:{doc_id}|level:{level}|n:{len(current)}"
        res: GroupResponse = self._cached_group(key, items_text, max_groups)

        children: List[SummaryChunk] = []
        for i, g in enumerate(res.groups):
            # derive span as min..max of member spans
            members = [current[i] for i in g.member_indices if 0 <= i < len(current)]
            if not members:
                continue
            start_page = min(m.span.start_page for m in members)
            end_page = max(m.span.end_page for m in members)
            start_char = min(m.span.start_char for m in members)
            end_char = max(m.span.end_char for m in members)
            child = SummaryChunk(
                id=f"grp:{uuid.uuid4() if self.use_uuid else i}|level{level}|doc:{doc_id}",
                title=g.title.strip()[:120] or "Group",
                summary=g.summary.strip(),
                span=Span(start_page=start_page, end_page=end_page, start_char=start_char, end_char=end_char),
                level=level,
            )
            children.append(child)
            self._ensure_node(doc_id, child)
            # persist parent/child adjacency between members, too
            self._persist_adjacency(doc_id, [m.id for m in members])

        # persist nodes for the new group level
        # for ch in children:
        #     self._ensure_node(doc_id, ch)

        return children

    def _force_concat(self, doc_id: str, chunks: Sequence[SummaryChunk], level: int) -> SummaryChunk:
        text = " ".join(f"{c.title}. {c.summary}" for c in chunks)
        # Don't call LLM; deterministic concatenation
        start_page = min(c.span.start_page for c in chunks)
        end_page = max(c.span.end_page for c in chunks)
        start_char = min(c.span.start_char for c in chunks)
        end_char = max(c.span.end_char for c in chunks)
        return SummaryChunk(
            id=f"final:{uuid.uuid4()}|doc{doc_id}"  if self.use_uuid else f"final|doc{doc_id}",
            title="Final Summary",
            summary=text[:2000],  # keep it bounded
            span=Span(start_page=start_page, end_page=end_page, start_char=start_char, end_char=end_char),
            level=level,
        )

    # ---------- persistence helpers ----------

    def _persist_leaf_nodes(self, doc_id: str, leaves: Sequence[LeafChunk]) -> List[Node]:
        nodes: List[Node] = []
        for i, leaf in enumerate(leaves, start=1):
            n = Node(
                id=f"leafnode:{uuid.uuid4() if self.use_uuid else i}",
                label=f"raw_text_chunk {i}",
                type="entity",
                summary=leaf.text,
                references=[
                    self._ref(doc_id, leaf.span, snippet=leaf.text[:160])
                ],
                doc_id=doc_id,
                properties={
                    "level": -1,
                    "source_leaf_id": leaf.id,
                },
                # embedding = self.engine._ef(leaf.text)[0]
            )
            if not self.engine._exists_node(n.id):
                self.engine.add_node(n, doc_id=doc_id)
            nodes.append(n)
        return nodes

    def _as_node(self, doc_id: str, ch: SummaryChunk) -> Node:
        label = f"summary:{ch.title}"
        
        return Node(
            id=f"{ch.id}",  # stable per chunk
            label=label,
            type="entity",
            summary=ch.summary,
            references=[self._ref(doc_id, ch.span, snippet=ch.summary[:160])],
            doc_id=doc_id,
            properties={"level": ch.level},
            # embedding=self.engine._ef(f"{label}: {ch.summary}")[0]
        )

    def _ensure_node(self, doc_id: str, ch: SummaryChunk) -> str:
        nid = ch.id #f"node:{ch.id}"
        if not self.engine._exists_node(nid):
            self.engine.add_node(self._as_node(doc_id, ch), doc_id=doc_id)
        return nid

    def _persist_layer(self, doc_id: str, *, parents, children):
        # Ensure children exist first
        self._persist_micro_chunks_as_nodes(doc_id, children)

        # Ensure parents exist (accept Node or SummaryChunk)
        parent_ids = []
        for p in parents:
            if isinstance(p, SummaryChunk):
                self._ensure_node(doc_id, p)
                parent_ids.append(self._as_node(doc_id, p).id)
            else:  # Node
                if not self.engine._exists_node(p.id):
                    self.engine.add_node(p, doc_id=doc_id)
                parent_ids.append(p.id)

        # Now it’s safe to create edges: parent -(summarizes)-> child, child -(details)-> parent
        child_ids = [self._as_node(doc_id, c).id for c in children]
        for cid in child_ids:
            for pid in parent_ids:
                self._bi_edge(doc_id, src=pid, tgt=cid, relation=REL_SUMMARIZES, reverse_relation=REL_DETAILS)

        # Sibling order on children (precedes/after)
        self._persist_adjacency(doc_id, child_ids)
    def _persist_micro_chunks_as_nodes(self, doc_id: str, micro_chunks: list[SummaryChunk]) -> list[str]:
        """
        Persist each micro-chunk as a Node (type='summary_chunk'), using the same IDs as the SummaryChunk objects.
        Returns the list of node IDs (identical to the SummaryChunk ids).
        """
        node_ids: list[str] = []
        for ch in micro_chunks:
            # Build a reference carrying the span for node_docs indexing
            ref = ReferenceSession(
                doc_id = doc_id,
                collection_page_url=f"document_collection/{doc_id}",
                document_page_url=f"document/{doc_id}",
                start_page=ch.span.start_page,
                end_page=ch.span.end_page,
                start_char=ch.span.start_char,
                end_char=ch.span.end_char,
                snippet=(ch.summary[:160] if ch.summary else None),
            )

            n = Node(
                id=ch.id,                          # keep identical to SummaryChunk.id so edges line up
                label=ch.title,
                type="entity",
                summary=ch.summary,
                references=[ref],
                doc_id=doc_id,                     # will also be set by engine.add_node(doc_id=...) but harmless here
                # embedding = self.engine._ef(ch.summary)[0]
            )

            # Idempotent add (don’t recreate if present)
            if not self.engine._exists_node(n.id):
                self.engine.add_node(n, doc_id=doc_id)

            node_ids.append(n.id)

        return node_ids
    def _persist_adjacency(self, doc_id: str, ordered_node_ids: Sequence[str]):
        """
        Persist asymmetric sibling adjacency:
        a -(precedes)-> b
        b -(after)----> a
        """
        for a, b in zip(ordered_node_ids, ordered_node_ids[1:]):
            self._bi_edge(doc_id, src=a, tgt=b, relation=REL_PRECEDES, reverse_relation=REL_AFTER)
            
    def _bi_edge(self, doc_id: str, *, src: str, tgt: str, relation: str, reverse_relation: str):
        "this document graph does not enfoce multi headed edge"
        
        # forward
        e1 = Edge(
            id=f"edge:{uuid.uuid4() if self.use_uuid else str(f'({src}::{relation}::{tgt})')}",
            label = f"{src}::{relation}::{tgt}",
            relation=relation,
            source_ids=[src],
            target_ids=[tgt],
            type="relationship",
            summary=f"{relation}: {src} → {tgt}", source_edge_ids = [], target_edge_ids = [],
            references=[self._ref(doc_id, Span(start_page=1, end_page=1, start_char=0, end_char=0), snippet=None)],
            doc_id=doc_id,
        )
        # reverse (distinct relation name)
        e2 = Edge(
            id=f"edge:{uuid.uuid4() if self.use_uuid else str(f'({tgt}::{relation}::{src})')}",
            label = f"{src}::{relation}::{tgt}",
            relation=reverse_relation,
            source_ids=[tgt],
            target_ids=[src],
            type="relationship",
            summary=f"{reverse_relation}: {tgt} → {src}", source_edge_ids = [], target_edge_ids = [],
            references=[self._ref(doc_id, Span(start_page=1, end_page=1, start_char=0, end_char=0), snippet=None)],
            doc_id=doc_id,
        )
        if not self.engine._exists_edge(e1.id):
            self.engine.add_edge(e1, doc_id=doc_id)
        if not self.engine._exists_edge(e2.id):
            self.engine.add_edge(e2, doc_id=doc_id)

    def _ref(self, doc_id: str, span: Span, *, snippet: Optional[str]) -> ReferenceSession:
        return ReferenceSession(
            doc_id = doc_id,
            collection_page_url=f"document_collection/{doc_id}",
            document_page_url=f"document/{doc_id}",
            start_page=span.start_page,
            end_page=span.end_page,
            start_char=span.start_char,
            end_char=span.end_char,
            snippet=snippet,
        )

# -----------------------------
# High-level orchestrator
# -----------------------------

class PagewiseSummaryIngestor(BaseDocumentGraphIngestor):
    """
    Side-car summarization/organization ingester.

    It DOES NOT call any of the engine's LLM ingest paths.
    It only uses:
      - engine.add_document
      - engine.add_node
      - engine.add_edge

    Relationships stored (all bidirectional):
      - leaf/page ↔ summarizes ↔ micro-chunk (level 0)
      - chunk_i ↔ next_to ↔ chunk_{i+1} (for each layer)
      - parent-layer chunk(s) ↔ grouped_into ↔ higher-layer chunk
    """

    # Fully implemented in the base; left here for future extension hooks.
    pass
