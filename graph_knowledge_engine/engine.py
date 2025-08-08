from typing import List, Optional, Dict, Any, Tuple
from chromadb import Client
from chromadb.config import Settings
from .models import (
    Node,
    Edge,
    Document,
    Domain,
    ReferenceSession,
    LLMGraphExtraction,
    LLMNode,
    LLMEdge,
    AdjudicationCandidate,
    AdjudicationQuestionCode,
    QUESTION_KEY,
    QUESTION_DESC,
    AdjudicationVerdict,
    LLMMergeAdjudication,
)
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import uuid
from joblib import Memory
import json, re, uuid
from typing import Optional, List
from .models import ReferenceSession, MentionVerification, LLMGraphExtraction, LLMNode, LLMEdge, Node, Edge, Document

_DOC_URL = "document/{doc_id}"

def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

def _json_or_none(v):
    return None if v is None else json.dumps(v)

def _default_verification(note: str = "fallback span") -> MentionVerification:
    return MentionVerification(method="heuristic", is_verified=False, score=None, notes=note)

def _default_ref(doc_id: str, snippet: Optional[str] = None) -> ReferenceSession:
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=_DOC_URL.format(doc_id=doc_id),
        start_page=1, end_page=1, start_char=0, end_char=0,
        snippet=snippet or None,
        verification=_default_verification()
    )

def _ensure_ref_span(ref: ReferenceSession, doc_id: str) -> ReferenceSession:
    # Make sure URLs point at this doc and spans are complete
    r = ref.model_copy(deep=True)
    if not r.collection_page_url:
        r.collection_page_url = f"document_collection/{doc_id}"
    if not r.document_page_url or str(doc_id) not in r.document_page_url:
        r.document_page_url = _DOC_URL.format(doc_id=doc_id)
    # Fill span if missing/bad
    if r.end_page < r.start_page:
        r.end_page = r.start_page
    if r.start_page == r.end_page and r.end_char < r.start_char:
        r.end_char = r.start_char
    if r.start_page is None or r.end_page is None:
        r.start_page, r.end_page = 1, 1
    if r.start_char is None or r.end_char is None:
        r.start_char, r.end_char = 0, 0
    # Default verification if absent
    if r.verification is None:
        r.verification = _default_verification("no explicit verification from LLM")
    return r

def _normalize_refs(refs: Optional[List[ReferenceSession]], doc_id: str, fallback_snippet: Optional[str]) -> List[ReferenceSession]:
    if not refs or len(refs) == 0:
        return [_default_ref(doc_id, snippet=fallback_snippet)]
    return [_ensure_ref_span(ref, doc_id) for ref in refs]
# Simple on-disk cache dir (optional)
memory = Memory(location=os.path.join(".cache", "my_cache"), verbose=0)

import itertools

def build_aliases(node_ids, edge_ids):
    node_aliases = {rid: f"N{i}" for i, rid in enumerate(node_ids, start=1)}
    edge_aliases = {rid: f"E{i}" for i, rid in enumerate(edge_ids, start=1)}
    alias_for_real = {**node_aliases, **edge_aliases}
    real_for_alias = {v: k for k, v in alias_for_real.items()}
    return alias_for_real, real_for_alias

def aliasify_graph(nodes, edges, alias_for_real):
    """Return shallow copies with ids replaced by aliases for prompt."""
    def a(rid): return alias_for_real.get(rid, rid)  # fallback: leave as-is
    aliased_nodes = [
        {
            "id": a(n["id"]),
            "label": n["label"],
            "type": n["type"],
            "summary": n.get("summary",""),
            # include minimal fields the LLM needs
        }
        for n in nodes
    ]
    aliased_edges = [
        {
            "id": a(e["id"]),
            "relation": e["relation"],
            "source_ids": [a(s) for s in e.get("source_ids", [])],
            "target_ids": [a(t) for t in e.get("target_ids", [])],
        }
        for e in edges
    ]
    return aliased_nodes, aliased_edges

def de_alias_ids(llm_result, real_for_alias):
    """Translate LLM aliases back to real UUIDs in-place."""
    def r(a): return real_for_alias.get(a, a)
    for n in llm_result.nodes:
        if n.id: n.id = r(n.id)
    for e in llm_result.edges:
        if e.id: e.id = r(e.id)
        e.source_ids = [r(x) for x in e.source_ids]
        e.target_ids = [r(x) for x in e.target_ids]
    return llm_result
from dataclasses import dataclass, field

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def uuid_to_base62(u: str) -> str:
    n = int(u.replace("-", ""), 16)
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 62)
        out.append(ALPHABET[r])
    return "".join(reversed(out))

def base62_to_uuid(s: str) -> str:
    n = 0
    for ch in s:
        n = n * 62 + ALPHABET.index(ch)
    hex128 = f"{n:032x}"
    return f"{hex128[0:8]}-{hex128[8:12]}-{hex128[12:16]}-{hex128[16:20]}-{hex128[20:]}"

@dataclass
class AliasBook:
    """Stable per-session alias book. Append-only for cache friendliness."""
    next_n: int = 1
    next_e: int = 1
    real_to_alias: dict = field(default_factory=dict)  # real_id -> alias "N#"/"E#"
    alias_to_real: dict = field(default_factory=dict)  # alias -> real_id

    def alias_for_node(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"N{self.next_n}"
        self.next_n += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def alias_for_edge(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"E{self.next_e}"
        self.next_e += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def assign_for_sets(self, node_ids: list[str], edge_ids: list[str]):
        for rid in node_ids:
            self.alias_for_node(rid)
        for rid in edge_ids:
            self.alias_for_edge(rid)

    def legend_delta(self, node_ids: list[str], edge_ids: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return only (real_id, alias) pairs that are NEW since last turn."""
        new_nodes, new_edges = [], []
        for rid in node_ids:
            if rid not in self.real_to_alias:
                new_nodes.append((rid, self.alias_for_node(rid)))
        for rid in edge_ids:
            if rid not in self.real_to_alias:
                new_edges.append((rid, self.alias_for_edge(rid)))
        return new_nodes, new_edges

# strategy toggle
# "session_alias" -> N#/E# with session-stable AliasBook (+ delta legend)
# "base62"        -> N~<22ch> / E~<22ch> (no legend, fully deterministic)
ID_STRATEGY = "session_alias"  # or "base62"

class GraphKnowledgeEngine:
    """High-level orchestration for extracting, storing, and adjudicating knowledge graph data."""

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys whose values are None. ChromaDB metadata rejects None values."""
        return _strip_none(metadata) #{k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _json_or_none(obj: Any) -> Optional[str]:
        return json.dumps(obj) if obj is not None else None
    
    def _select_doc_context(self, doc_id: str, max_nodes: int = 200, max_edges: int = 400):
        nodes = self.node_collection.get(where={"doc_id": doc_id}, include=["documents"])
        edges = self.edge_collection.get(where={"doc_id": doc_id}, include=["documents"])

        node_items = []
        for i, (nid, ndoc) in enumerate(zip(nodes.get("ids", []) or [], nodes.get("documents", []) or [])):
            if i >= max_nodes:
                break
            n = Node.model_validate_json(ndoc)
            node_items.append({"id": nid, "label": n.label, "type": n.type, "summary": n.summary})

        edge_items = []
        for i, (eid, edoc) in enumerate(zip(edges.get("ids", []) or [], edges.get("documents", []) or [])):
            if i >= max_edges:
                break
            e = Edge.model_validate_json(edoc)
            edge_items.append({"id": eid, "relation": e.relation, "source_ids": e.source_ids or [], "target_ids": e.target_ids or []})

        return node_items, edge_items
    def _aliasify_for_prompt(self, doc_id: str, ctx_nodes: list[dict], ctx_edges: list[dict]):
        """Return aliased nodes/edges + prompt strings, using configured ID_STRATEGY."""
        if ID_STRATEGY == "base62":
            # Deterministic short IDs, no legend needed
            aliased_nodes = []
            for n in ctx_nodes:
                aliased_nodes.append({
                    "id": f"N~{uuid_to_base62(n['id'])}",
                    "label": n["label"], "type": n["type"], "summary": n.get("summary", "")
                })
            aliased_edges = []
            for e in ctx_edges:
                aliased_edges.append({
                    "id": f"E~{uuid_to_base62(e['id'])}",
                    "relation": e["relation"],
                    "source_ids": [f"N~{uuid_to_base62(s)}" for s in e.get("source_ids", [])],
                    "target_ids": [f"N~{uuid_to_base62(t)}" for t in e.get("target_ids", [])],
                })
            return aliased_nodes, aliased_edges, "Node aliases: (implicit base62)", "Edge aliases: (implicit base62)"

        # session_alias (cache-friendly with delta legend)
        book = self._alias_book(doc_id)
        node_ids = [n["id"] for n in ctx_nodes]
        edge_ids = [e["id"] for e in ctx_edges]
        # Only list *new* aliases to keep prompts stable
        new_nodes, new_edges = book.legend_delta(node_ids, edge_ids)

        def a_node(x): return book.real_to_alias[x]
        def a_edge(x): return book.real_to_alias[x]

        aliased_nodes = [{"id": a_node(n["id"]), "label": n["label"], "type": n["type"], "summary": n.get("summary", "")} for n in ctx_nodes]
        aliased_edges = [{"id": a_edge(e["id"]), "relation": e["relation"],
                        "source_ids": [a_node(s) for s in e.get("source_ids", [])],
                        "target_ids": [a_node(t) for t in e.get("target_ids", [])]} for e in ctx_edges]

        # Build tiny delta legend strings (only new aliases)
        if new_nodes:
            lines = [f"- {book.real_to_alias[rid]}: {next(n for n in ctx_nodes if n['id']==rid)['label']}" for rid, _ in new_nodes]
            nodes_str = "New node aliases:\n" + "\n".join(lines)
        else:
            nodes_str = "New node aliases: (none)"

        if new_edges:
            lines = []
            for rid, _ in new_edges:
                e = next(e for e in ctx_edges if e["id"] == rid)
                lines.append(f"- {book.real_to_alias[rid]}: {e['relation']}")
            edges_str = "New edge aliases:\n" + "\n".join(lines)
        else:
            edges_str = "New edge aliases: (none)"

        return aliased_nodes, aliased_edges, nodes_str, edges_str
    def _extract_graph_with_llm_aliases(self, content: str, alias_nodes_str: str, alias_edges_str: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an expert knowledge graph extractor. "
            "Extract entities and relationships as nodes and edges in a hypergraph.\n\n"
            "Rules:\n"
            "1) When referring to existing items, use ONLY the given aliases.\n"
            "2) If creating new items, omit their id.\n"
            "3) Each node/edge MUST include at least one ReferenceSession with spans.\n"
            "4) Do not invent aliases; use the provided ones only."),
            ("human",
            "Aliases (delta for this turn):\n{alias_nodes}\n\n{alias_edges}\n\n"
            "Document chunk:\n{document}\n\n"
            "Return only the structured JSON for the schema.")
        ])
        chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
        result = chain.invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content})
        return result.get("raw"), result.get("parsed"), result.get("parsing_error")
    def _de_alias_ids_in_result(self, doc_id: str, parsed: LLMGraphExtraction) -> LLMGraphExtraction:
        """Map aliases back to real UUIDs according to strategy."""
        if ID_STRATEGY == "base62":
            def r(s: str | None):
                if not s:
                    return s
                if s.startswith("N~"):
                    return base62_to_uuid(s[2:])
                if s.startswith("E~"):
                    return base62_to_uuid(s[2:])
                return s
        else:
            book = self._alias_book(doc_id)
            def r(s: str | None):
                if not s:
                    return s
                return book.alias_to_real.get(s, s)

        # mutate copy
        for n in parsed.nodes:
            if n.id: n.id = r(n.id)
        for e in parsed.edges:
            if e.id: e.id = r(e.id)
            e.source_ids = [r(x) for x in e.source_ids]
            e.target_ids = [r(x) for x in e.target_ids]
        return parsed

    def _alias_legend_strings(self, aliased_nodes, aliased_edges):
        """Tiny text blocks for the prompt, to guide the model to use aliases only."""
        if aliased_nodes:
            node_lines = [f"- {n['id']}: {n['label']} [{n['type']}] — {n.get('summary','')}" for n in aliased_nodes]
            nodes_str = "Node aliases:\n" + "\n".join(node_lines)
        else:
            nodes_str = "Node aliases: (none)"

        if aliased_edges:
            def fmt_ids(xs): return ", ".join(xs)
            edge_lines = [
                f"- {e['id']}: {e['relation']} — src[{fmt_ids(e.get('source_ids', []))}] → tgt[{fmt_ids(e.get('target_ids', []))}]"
                for e in aliased_edges
            ]
            edges_str = "Edge aliases:\n" + "\n".join(edge_lines)
        else:
            edges_str = "Edge aliases: (none)"

        return nodes_str, edges_str
    def _alias_book(self, key: str) -> AliasBook:
        if key not in self._alias_books:
            self._alias_books[key] = AliasBook()
        return self._alias_books[key]
    # ----------------------------
    # Init
    # ----------------------------
    def __init__(self, persist_directory: Optional[str] = None):
        load_dotenv()
        self._alias_books: dict[str, AliasBook] = {}
        self.chroma_client = Client(
            Settings(
                is_persistent=True,
                persist_directory=persist_directory or "./chroma_db",
                anonymized_telemetry=False,
            )
        )
        self.node_collection = self.chroma_client.get_or_create_collection("nodes")
        self.edge_collection = self.chroma_client.get_or_create_collection("edges")
        self.edge_endpoints_collection = self.chroma_client.get_or_create_collection("edge_endpoints")
        self.document_collection = self.chroma_client.get_or_create_collection("documents")
        self.domain_collection = self.chroma_client.get_or_create_collection("domains")

        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=12000,
            openai_api_type="azure",
        )

    # ----------------------------
    # Chroma adders
    # ----------------------------
    def add_node(self, node: Node, doc_id: Optional[str] = None):
        node.doc_id = doc_id
        self.node_collection.add(
            ids=[node.id],
            documents=[node.model_dump_json()],
            embeddings=[node.embedding] if node.embedding else None,
            metadatas=[_strip_none({
                "doc_id": doc_id,
                "label": node.label,
                "type": node.type,
                "summary": node.summary,
                "domain_id": node.domain_id,
                "canonical_entity_id": node.canonical_entity_id,
                "properties": _json_or_none(node.properties),
                "references": _json_or_none([r.model_dump() for r in node.references]),
            })],
        )
    def add_edge(self, edge: Edge, doc_id: Optional[str] = None):
        edge.doc_id = doc_id
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[_strip_none({
                "doc_id": doc_id,
                "relation": edge.relation,
                "source_ids": _json_or_none(edge.source_ids),
                "target_ids": _json_or_none(edge.target_ids),
                "type": edge.type,
                "summary": edge.summary,
                "domain_id": edge.domain_id,
                "canonical_entity_id": edge.canonical_entity_id,
                "properties": _json_or_none(edge.properties),
                "references": _json_or_none([r.model_dump() for r in edge.references]),
            })],
        )

        # write edge_endpoints fanout (strip None so Chroma doesn’t choke)
        ep_ids, ep_docs, ep_meta = [], [], []
        for role, node_ids in (("src", edge.source_ids or []), ("tgt", edge.target_ids or [])):
            for nid in node_ids:
                eid = f"{edge.id}::{role}::{nid}"
                m = _strip_none({
                    "id": eid,
                    "edge_id": edge.id,
                    "node_id": nid,
                    "role": role,
                    "relation": edge.relation,
                    "doc_id": doc_id,
                })
                ep_ids.append(eid)
                ep_docs.append(json.dumps(m))
                ep_meta.append(m)

        if ep_ids:
            self.edge_endpoints_collection.add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_meta,
            )
    def add_document(self, document: Document):
        self.document_collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[_strip_none({
                "doc_id": document.id,  # <— critical
                "type": document.type,
                "metadata": _json_or_none(document.metadata),
                "domain_id": document.domain_id,
                "processed": document.processed,
            })],
        )

    def add_domain(self, domain: Domain):
        self.domain_collection.add(
            ids=[domain.id],
            documents=[domain.model_dump_json()],
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "name": domain.name,
                        "description": domain.description,
                    }
                )
            ],
        )
    # ----------------------------
    # helpers for rollback
    # ----------------------------
    def _edge_ids_by_doc(self, document_id: str) -> list[str]:
        eps = self.edge_endpoints_collection.get(where={"doc_id": document_id})
        if not eps["documents"]:
            return []
        return list({json.loads(doc)["edge_id"] for doc in eps["documents"]})

    def _delete_edges_by_ids(self, edge_ids: list[str]):
        if not edge_ids:
            return
        self.edge_collection.delete(ids=edge_ids)
        # also delete their endpoint rows
        # endpoints ids start with f"{edge_id}::"
        # we can delete via where on edge_id for simplicity:
        self.edge_endpoints_collection.delete(where={"edge_id": {"$in": edge_ids}})
    # ----------------------------
    # Vector queries
    # ----------------------------
    def vector_search_nodes(self, embedding: List[float], top_k: int = 5):
        return self.node_collection.query(query_embeddings=[embedding], n_results=top_k)

    def vector_search_edges(self, embedding: List[float], top_k: int = 5):
        return self.edge_collection.query(query_embeddings=[embedding], n_results=top_k)

    
    # def _extract_graph_with_llm_aliases(self, content: str, alias_nodes_str: str, alias_edges_str: str):
    #     """
    #     Ask the LLM to extract nodes/edges but require it to use ID aliases we provide.
    #     """
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system",
    #             "You are an expert knowledge graph extractor. "
    #             "Extract entities and relationships as nodes and edges in a hypergraph.\n\n"
    #             "RULES:\n"
    #             "1) When referring to any existing node/edge, use ONLY the aliases provided below.\n"
    #             "2) If you must create a new node/edge, omit its id (we will assign one).\n"
    #             "3) Each node/edge MUST include at least one ReferenceSession with start_page, end_page, start_char, end_char.\n"),
    #             ("human",
    #             "Use these aliases (do not invent new aliases):\n\n"
    #             "{alias_nodes}\n\n{alias_edges}\n\n"
    #             "Now extract from this content:\n{document}\n\nReturn only the structured JSON for the schema.")
    #         ]
    #     )
    #     chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
    #     result = chain.invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content})
    #     raw = result.get("raw") if isinstance(result, dict) else None
    #     parsed = result.get("parsed") if isinstance(result, dict) else result
    #     err = result.get("parsing_error") if isinstance(result, dict) else None
    #     return raw, parsed, err

    def _extract_graph_with_llm(self, content: str) -> Tuple[Any, Optional[LLMGraphExtraction], Optional[str]]:
        """Call LLM for structured extraction. Returns (raw, parsed, parsing_error)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert knowledge graph extractor. Given a document, extract entities and relationships as nodes and edges in a hypergraph.\n"
                    "For each node/edge include: label, type ('entity' or 'relationship'), and a concise 'summary'.\n"
                    "Each node/edge MUST include at least one ReferenceSession with start_page, end_page, start_char, end_char."
                ),
                ("human", "Document:\n{document}\n\nReturn only the structured JSON as specified."),
                
            ]
        )
        chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
        result = chain.invoke({"document": content})
        # result is a dict when include_raw=True
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        err = result.get("parsing_error") if isinstance(result, dict) else None
        return raw, parsed, err

    def _ingest_text_with_llm(
        self,
        *,
        doc_id: str,
        content: str,
        auto_adjudicate: bool = False,
        context_nodes_limit: int = 200,
        context_edges_limit: int = 400,
    ):
        """
        Shared ingestion path:
        - Build small doc-scoped context
        - Alias IDs to save tokens
        - Instruct LLM to use aliases
        - Parse, de-alias, normalize refs, and store
        - Optionally auto-adjudicate (within this document)
        """
        # 1) Context + aliases
        ctx_nodes, ctx_edges = self._select_doc_context(doc_id, context_nodes_limit, context_edges_limit)
        alias_for_real, real_for_alias = build_aliases([n["id"] for n in ctx_nodes], [e["id"] for e in ctx_edges])
        aliased_nodes, aliased_edges = aliasify_graph(ctx_nodes, ctx_edges, alias_for_real)
        alias_nodes_str, alias_edges_str = self._alias_legend_strings(aliased_nodes, aliased_edges)

        # 2) Extract with aliases
        raw, parsed, error = self._extract_graph_with_llm_aliases(content, alias_nodes_str, alias_edges_str)
        if error:
            raise ValueError(f"LLM parsing error: {error}")
        if not isinstance(parsed, LLMGraphExtraction):
            parsed = LLMGraphExtraction.model_validate(parsed)

        # 3) De-alias the result
        parsed = de_alias_ids(parsed, real_for_alias)

        # 4) Normalize refs and persist
        fallback_snip = (content[:160] + "…") if content else None
        nodes_added, edges_added = 0, 0
        for ln in parsed.nodes:
            nid = ln.id or str(uuid.uuid1())
            refs = _normalize_refs(ln.references, doc_id, fallback_snip)
            node = Node(
                id=nid,
                label=ln.label,
                type=ln.type,
                summary=ln.summary,
                domain_id=ln.domain_id,
                canonical_entity_id=ln.canonical_entity_id,
                properties=ln.properties,
                references=refs,
                embedding=None,
                doc_id=doc_id,
            )
            self.add_node(node, doc_id=doc_id)
            nodes_added += 1

        for le in parsed.edges:
            eid = le.id or str(uuid.uuid1())
            refs = _normalize_refs(le.references, doc_id, fallback_snip)
            edge = Edge(
                id=eid,
                label=le.label,
                type=le.type,
                summary=le.summary,
                domain_id=le.domain_id,
                canonical_entity_id=le.canonical_entity_id,
                properties=le.properties,
                references=refs,
                source_ids=le.source_ids,
                target_ids=le.target_ids,
                relation=le.relation,
                embedding=None,
                doc_id=doc_id,
            )
            self.add_edge(edge, doc_id=doc_id)
            edges_added += 1

        # 5) Optional within-doc auto-adjudication
        if auto_adjudicate:
            # simple within-doc candidate generation: label buckets + type match
            new_nodes = self.node_collection.get(where={"doc_id": doc_id}, include=["ids", "documents"])
            by_key = {}
            for nid, ndoc in zip(new_nodes.get("ids", []), new_nodes.get("documents", [])):
                n = Node.model_validate_json(ndoc)
                key = (n.type, n.label.strip().lower())
                by_key.setdefault(key, []).append(n)

            pairs = []
            for _, bucket in by_key.items():
                if len(bucket) > 1:
                    # naive pairwise; you can prune by summary similarity later
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            pairs.append((bucket[i], bucket[j]))

            if pairs:
                verdicts, _ = self.batch_adjudicate_merges(pairs)
                # commit only positive merges
                for pair, out in zip(pairs, verdicts):
                    v = out.verdict if hasattr(out, "verdict") else out  # LC variant safety
                    if v.same_entity:
                        try:
                            self.commit_merge(pair[0], pair[1], v)
                        except Exception:
                            pass

        return {"nodes_added": nodes_added, "edges_added": edges_added, "raw": raw}

    def prune_node_from_edges(self, node_id: str):
        # find edges via endpoints (fast, indexed)
        eps = self.edge_endpoints_collection.get(where={"node_id": node_id}, include=["documents"])
        if not eps["ids"]:
            return {"deleted_edges": set(), "updated_edges": set()}

        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps["documents"]})
        edges = self.edge_collection.get(ids=edge_ids, include=["documents", "metadatas"])
        updated_edge_ids = set()
        removed_edge_ids = set()
        deleted = updated = 0
        for eid, edoc, meta in zip(edges["ids"], edges["documents"], edges["metadatas"]):
            e = Edge.model_validate_json(edoc)
            new_src = [x for x in (e.source_ids or []) if x != node_id]
            new_tgt = [x for x in (e.target_ids or []) if x != node_id]

            if not new_src or not new_tgt:
                # delete the whole edge + its endpoint rows
                self.edge_collection.delete(ids=[eid])
                self.edge_endpoints_collection.delete(where={"edge_id": eid})
                deleted += 1
                removed_edge_ids.update([eid])
            else:
                e.source_ids, e.target_ids = new_src, new_tgt
                self.edge_collection.update(
                    ids=[eid],
                    documents=[e.model_dump_json()],
                    metadatas=[_strip_none({
                        "doc_id": meta.get("doc_id"),
                        "relation": e.relation,
                        "source_ids": _json_or_none(e.source_ids),
                        "target_ids": _json_or_none(e.target_ids),
                        "type": e.type,
                        "summary": e.summary,
                        "domain_id": e.domain_id,
                        "canonical_entity_id": e.canonical_entity_id,
                        "properties": _json_or_none(e.properties),
                        "references": _json_or_none([ref.model_dump() for ref in (e.references or [])]),
                    })],
                )
                updated_edge_ids.add([eid])
                # remove only the touched endpoint rows
                self.edge_endpoints_collection.delete(where={"$and": [{"edge_id": eid}, {"node_id": node_id}]})
                updated += 1

        return {"deleted_edges": removed_edge_ids, 
                "updated_edges": updated_edge_ids - removed_edge_ids} # a multiedge may update first and then deleted
    def verify_mentions_for_doc(self, document_id: str, method: str = "levenshtein"):
        """Run a verifier over all nodes/edges of a doc and update their references.verification."""
        # Fetch nodes for this doc
        nodes = self.node_collection.get(where={"doc_id": document_id}, include=["ids", "documents"])
        for node_id, doc_json in zip(nodes["ids"], nodes["documents"]):
            node = Node.model_validate_json(doc_json)
            updated = False
            for ref in node.references:
                if ref.verification and ref.verification.is_verified is not False:
                    continue
                # TODO: implement your method here against source text
                ref.verification = MentionVerification(method=method, is_verified=False, notes="not implemented")
                updated = True
            if updated:
                self.node_collection.update(
                    ids=[node_id],
                    documents=[node.model_dump_json()],
                )
    def rollback_document(self, document_id: str):
        # 1) nodes by flat doc_id
        nodes = self.node_collection.get(where={"doc_id": document_id})
        node_ids = nodes.get("ids", []) or []

        # 2) prune each node from edges
        deleted_edges = set()
        updated_edges = set()
        deleted_edges_cnt = updated_edges_cnt = 0
        for nid in node_ids:
            res = self.prune_node_from_edges(nid)
            deleted_edges_cnt += len(res["deleted_edges"])
            updated_edges_cnt += len(res["updated_edges"])
            deleted_edges.update(res["deleted_edges"])
            updated_edges.update(res["updated_edges"])
        
        # 3) delete edges explicitly created by this document via endpoints table
        eps = self.edge_endpoints_collection.get(where={"doc_id": document_id})
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps.get("documents", [])})
        if edge_ids:
            self.edge_collection.delete(ids=edge_ids)
            self.edge_endpoints_collection.delete(where={"doc_id": document_id})
            deleted_edges.update(edge_ids)
        updated_edges = updated_edges - deleted_edges
        # 4) delete nodes and the document
        if node_ids:
            self.node_collection.delete(ids=node_ids)
        self.document_collection.delete(where={"doc_id": document_id})

        return {
            "rolled_back_doc_id": document_id,
            "updated_edge_ids": list(updated_edges),
            "deleted_edge_ids": list(deleted_edges),
            "deleted_node_ids": node_ids,
            "deleted_nodes": len(node_ids),
            "deleted_edges": len(deleted_edges),
            "updated_edges": len(updated_edges),
        }

    def rollback_many_documents(self, document_ids: list[str]):
        totals = {"deleted_nodes": 0, "deleted_edges": 0, "updated_edges": 0, "deleted_docs": 0}
        for did in document_ids:
            res = self.rollback_document(did)
            totals["deleted_docs"] += 1
            totals["deleted_nodes"] += len(res["deleted_node_ids"])
            totals["deleted_edges"] += len(res["deleted_edge_ids"])
            totals["updated_edges"] += res["updated_edges"]
        return totals
    # ----------------------------
    # Adjudication (LLM-assisted merge decision)
    # ----------------------------
    def adjudicate_merge(self, left: Node, right: Node) -> AdjudicationVerdict:
        """Use the LLM to decide if two nodes are the SAME real-world entity."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful adjudicator. Decide if two nodes refer to the SAME real-world entity.\n"
                    "Be conservative: only return true if confident. Return a structured JSON verdict.",
                ),
                ("human", "Left:\n{left}\n\nRight:\n{right}\n\nReturn only the structured JSON."),
            ]
        )
        chain = prompt | self.llm.with_structured_output(LLMMergeAdjudication)
        result: LLMMergeAdjudication = chain.invoke(
            {"left": left.model_dump(), "right": right.model_dump()}
        )
        return result.verdict

    def commit_merge(self, left: Node, right: Node, verdict: AdjudicationVerdict) -> str:
        """
        Apply a positive adjudication by assigning/propagating a canonical_entity_id
        and recording a `same_as` edge with provenance.
        Returns the canonical id used.
        """
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")

        canonical_id = verdict.canonical_entity_id or (left.canonical_entity_id or right.canonical_entity_id)
        if not canonical_id:
            canonical_id = str(uuid.uuid1())

        # Update in-memory nodes (caller is responsible for persisting updates to DB)
        left.canonical_entity_id = canonical_id
        right.canonical_entity_id = canonical_id

        # Record a same_as edge for auditability
        same_as = Edge(
            id=str(uuid.uuid1()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "Adjudicated same entity",
            domain_id=None,
            relation="same_as",
            source_ids=[left.id],
            target_ids=[right.id],
            properties={"confidence": verdict.confidence},
            references=None,
        )
        self.add_edge(same_as)
        return canonical_id

    def generate_merge_candidates(self, new_node: Node, top_k: int = 5, similarity_threshold: float = 0.85) -> List[Tuple[Node, Node]]:
        """
        Given a new node, find likely duplicates in Chroma for adjudication.
        Returns a list of (existing_node, new_node) pairs.
        """
        if not new_node.embedding:
            # Skip vector search if no embedding
            return []

        results = self.node_collection.query(
            query_embeddings=[new_node.embedding],
            n_results=top_k
        )

        candidates = []
        for idx, doc_json in enumerate(results["documents"][0]):
            score = results["distances"][0][idx]
            if score >= similarity_threshold:
                existing_node = Node(**json.loads(doc_json))
                # Don't match against itself
                if existing_node.id != new_node.id:
                    candidates.append((existing_node, new_node))
        return candidates
    
    def batch_adjudicate_merges(self, pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY):
        if not pairs:
            return []

        mapping_table = [
            {"code": int(code), "key": QUESTION_KEY[code], "description": QUESTION_DESC[code]}
            for code in AdjudicationQuestionCode
        ]

        adjudication_inputs = [
            {
                "left": left.model_dump(),
                "right": right.model_dump(),
                "question_code": int(question_code),
            }
            for left, right in pairs
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You adjudicate candidate pairs. Use the mapping table to interpret question_code. "
            "Return only the structured JSON per schema."),
            ("human",
            "Mapping table:\n{mapping}\n\nPairs:\n{pairs}")
        ])

        chain = prompt | self.llm.with_structured_output(LLMMergeAdjudication, many=True)
        results = chain.invoke({"mapping": mapping_table, "pairs": adjudication_inputs})

        # (Optional) convert code → key on the backend if you store it
        qkey = QUESTION_KEY[AdjudicationQuestionCode(question_code)]
        return results, qkey
        
    def ingest_document_with_llm(self, document: Document):
        self.add_document(document)
        res = self._ingest_text_with_llm(
            doc_id=document.id,
            content=document.content,
            auto_adjudicate=False,  # leave adjudication choice to user at document level
        )
        return {"document_id": document.id, **res}

    def add_page(self, *, document_id: str, page_text: str, page_number: int, auto_adjudicate: bool = True):
        """
        Ingest a single page of an existing document.
        - Reuses doc-scoped context with aliases (cheap tokens)
        - Stores nodes/edges tagged with doc_id
        - Auto-adjudicates within the document by default
        """
        if not page_text or not isinstance(page_text, str):
            return {"document_id": document_id, "nodes_added": 0, "edges_added": 0}

        # Optionally, you could attach a page-level Document record (virtual).
        # For now we just ingest content with doc_id and normalize refs as usual.
        res = self._ingest_text_with_llm(
            doc_id=document_id,
            content=page_text,
            auto_adjudicate=auto_adjudicate,
        )
        # If you want to force mention spans to that page when missing,
        # you could post-process the brand-new nodes/edges and ensure refs' start_page/end_page = page_number.
        return {"document_id": document_id, "page": page_number, **res}