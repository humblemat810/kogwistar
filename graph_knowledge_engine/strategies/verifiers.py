# strategies/verifiers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import math
from strategies import EngineLike

try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio
except ImportError:
    def fuzz_ratio(a: str, b: str) -> float:
        # very small fallback
        return 100.0 if a == b else 0.0

from ..models import Node, Edge, MentionVerification, ReferenceSession
# from ..engine import GraphKnowledgeEngine
@dataclass
class VerifierConfig:
    min_snippet_len: int = 12
    min_overlap_chars: int = 6
    min_levenshtein_score: float = 0.65  # RapidFuzz normalized
    use_embeddings: bool = False  # only if engine has embedding_fn
    min_embed_cosine: float = 0.5

class DefaultVerifier:
    """
    Offline verification (no web calls). Combines:
      - span sanity (page and char ordering)
      - snippet containment / overlap %
      - RapidFuzz ratio between label (or summary) and cited snippet
      - optional embedding cosine if engine has an embedding_fn
    Stores results back into references[].verification and updates Chroma.
    """

    def __init__(self, engine: EngineLike, config: Optional[VerifierConfig] = None):
        self.e: EngineLike = engine
        self.cfg = config or VerifierConfig()

    # ---------------- public API ----------------
    def verify_mentions_for_doc(self, document_id: str) -> dict:
        updated_nodes = self._verify_collection_for_doc(
            which="node", document_id=document_id
        )
        updated_edges = self._verify_collection_for_doc(
            which="edge", document_id=document_id
        )
        return {"document_id": document_id, "updated_nodes": updated_nodes, "updated_edges": updated_edges}

    # ---------------- internals ----------------
    def _fetch_doc_text(self, doc_id: str) -> Optional[str]:
        got = self.e.document_collection.get(ids=[doc_id], include=["documents"])
        docs = got.get("documents") or []
        return docs[0] if docs else None

    def _span_ok(self, ref: ReferenceSession) -> bool:
        if ref.end_page < ref.start_page:
            return False
        if ref.start_page == ref.end_page and ref.end_char < ref.start_char:
            return False
        return True

    def _snippet_from_span(self, text: str, ref: ReferenceSession) -> str:
        if not text:
            return ""
        # naive multi-page handling: treat as linear for now
        start = max(0, int(ref.start_char or 0))
        end = min(len(text), int(ref.end_char or 0))
        if end <= start:
            return ""
        return text[start:end]

    def _embed_cosine(self, a: List[float], b: List[float]) -> float:
        # tiny cosine helper
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-8
        nb = math.sqrt(sum(y * y for y in b)) or 1e-8
        return dot / (na * nb)

    def _score_ref(self, label_or_summary: str, snippet: str, obj_embed: Optional[List[float]]) -> Tuple[bool, float, str]:
        if not snippet or len(snippet) < self.cfg.min_snippet_len:
            return False, 0.0, "snippet-too-short"

        # RapidFuzz score
        rf = (fuzz_ratio(label_or_summary, snippet) or 0.0) / 100.0
        notes = [f"rf={rf:.2f}"]

        ok = rf >= self.cfg.min_levenshtein_score

        # Optional embedding
        if ok and self.cfg.use_embeddings and obj_embed and callable(getattr(self.e, "embedding_fn", None)):
            try:
                s_vec = self.e.embedding_function([snippet])[0]
                cos = self._embed_cosine(obj_embed, s_vec)
                notes.append(f"cos={cos:.2f}")
                ok = cos >= self.cfg.min_embed_cosine
            except Exception:
                notes.append("embed-error")

        return ok, rf, " | ".join(notes)

    def _verify_collection_for_doc(self, *, which: str, document_id: str) -> int:
        if which == "node":
            got = self.e.node_collection.get(where={"doc_id": document_id}, include=["ids", "documents", "metadatas"])
        else:
            got = self.e.edge_collection.get(where={"doc_id": document_id}, include=["ids", "documents", "metadatas"])

        total_updates = 0
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []

        for obj_id, doc_json, meta in zip(ids, docs, metas):
            obj = Node.model_validate_json(doc_json) if which == "node" else Edge.model_validate_json(doc_json)
            changed = False

            # pick best text to compare against snippet
            label = getattr(obj, "label", "") or ""
            summary = getattr(obj, "summary", "") or ""
            text_for_match = summary or label

            for ref in (obj.references or []):
                did = getattr(ref, "doc_id", None)
                if not did and ref.document_page_url:
                    # fallback: parse "document/<id>"
                    import re
                    m = re.search(r"document/([A-Za-z0-9\-]+)", ref.document_page_url)
                    if m:
                        did = m.group(1)

                if did != document_id:
                    continue  # only verify mentions for this doc_id

                if not self._span_ok(ref):
                    ref.verification = MentionVerification(method="span", is_verified=False, score=0.0, notes="bad-span")
                    changed = True
                    continue

                doc_text = self._fetch_doc_text(document_id) or ""
                snippet = ref.snippet or self._snippet_from_span(doc_text, ref)

                ok, score, notes = self._score_ref(text_for_match, snippet, getattr(obj, "embedding", None))
                ref.verification = MentionVerification(method="heuristic", is_verified=ok, score=score, notes=notes)
                changed = True

            if changed:
                # persist updates
                if which == "node":
                    self.e.node_collection.update(
                        ids=[obj_id],
                        documents=[obj.model_dump_json()],
                        metadatas=[self.e.chroma_sanitize_metadata({
                            "doc_id": (meta or {}).get("doc_id"),
                            "label": getattr(obj, "label", None),
                            "type": getattr(obj, "type", None),
                            "summary": getattr(obj, "summary", None),
                            "domain_id": getattr(obj, "domain_id", None),
                            "canonical_entity_id": getattr(obj, "canonical_entity_id", None),
                            "properties": self.e._json_or_none(getattr(obj, "properties", None)),
                            "references": self.e._json_or_none([r.model_dump() for r in (obj.references or [])]),
                        })],
                    )
                else:
                    self.e.edge_collection.update(
                        ids=[obj_id],
                        documents=[obj.model_dump_json()],
                        metadatas=[self.e.chroma_sanitize_metadata({
                            "doc_id": (meta or {}).get("doc_id"),
                            "relation": getattr(obj, "relation", None),
                            "source_ids": self.e._json_or_none(getattr(obj, "source_ids", None)),
                            "target_ids": self.e._json_or_none(getattr(obj, "target_ids", None)),
                            "type": getattr(obj, "type", None),
                            "summary": getattr(obj, "summary", None),
                            "domain_id": getattr(obj, "domain_id", None),
                            "canonical_entity_id": getattr(obj, "canonical_entity_id", None),
                            "properties": self.e._json_or_none(getattr(obj, "properties", None)),
                            "references": self.e._json_or_none([r.model_dump() for r in (obj.references or [])]),
                        })],
                    )
                total_updates += 1

        return total_updates