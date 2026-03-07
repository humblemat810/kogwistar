# strategies/verifiers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import math
from .types import EngineLike, Verifier
import json

try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    def fuzz_ratio(a: str, b: str) -> float:
        # very small fallback
        return 100.0 if a == b else 0.0

from ..engine_core.models import Node, Edge, MentionVerification, Span
# from ..engine_core.engine import GraphKnowledgeEngine
@dataclass
class VerifierConfig:
    min_excerpt_len: int = 12
    min_overlap_chars: int = 6
    min_levenshtein_score: float = 0.65  # RapidFuzz normalized
    use_embeddings: bool = False  # only if engine has embedding_fn
    min_embed_cosine: float = 0.5

class DefaultVerifier(Verifier):
    """
    Offline verification (no web calls). Combines:
      - span sanity (page and char ordering)
      - excerpt containment / overlap %
      - RapidFuzz ratio between label (or summary) and cited excerpt
      - optional embedding cosine if engine has an embedding_fn
    Stores results back into references[].verification and updates Chroma.
    """
    @staticmethod
    def _normalize(s: str) -> str:
        return " ".join((s or "").split()).strip().lower()
    

    @staticmethod
    def _slice_span(full_text: str, start: int, end: int) -> str:
        start = max(0, start or 0)
        end = max(start, end or start)
        end = min(len(full_text), end)
        return full_text[start:end]
    def _score_rapidfuzz(self, a: str, b: str) -> Optional[float]:
        if not _HAS_RAPIDFUZZ:
            return None
        a, b = self._normalize(a), self._normalize(b)
        if not a or not b:
            return None
        # token_set_ratio is robust to word order/noise; scale 0..100 -> 0..1
        return float(fuzz.token_set_ratio(a, b)) / 100.0    


    @staticmethod
    def _score_coverage(extracted: str, cited: str, min_ngram: int = 5) -> Optional[float]:
        """
        Percent of extracted characters covered by any n-gram (len>=min_ngram)
        that also appears in cited. Simple, fast lower bound for span support.
        """
        ex_norm = " ".join((extracted or "").split())
        ci_norm = " ".join((cited or "").split())
        if not ex_norm or not ci_norm:
            return None
        n = max(1, min_ngram)
        covered = [False] * len(ex_norm)
        # greedy n-gram cover: slide windows, mark matches
        for i in range(0, len(ex_norm) - n + 1):
            gram = ex_norm[i:i+n]
            if gram in ci_norm:
                for j in range(i, i+n):
                    covered[j] = True
        # expand matches (optional): if surrounded by covered, keep filling
        # (skip for speed; baseline is fine)
        total = len(ex_norm)
        hit = sum(1 for x in covered if x)
        return hit / total if total > 0 else None

    def _score_embedding(self, extracted: str, cited: str) -> float | None:
        u = self._embed_one(extracted or "")
        v = self._embed_one(cited or "")
        if u is None or v is None: 
            return None
        # cosine
        dot = sum(a*b for a, b in zip(u, v))
        nu = (sum(a*a for a in u)) ** 0.5
        nv = (sum(b*b for b in v)) ** 0.5
        return float((dot / (nu * nv))) if nu and nv else None

    def _ensemble(self, scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
        """Weighted average over available (non-None) scores."""
        num = 0.0
        den = 0.0
        for k, w in weights.items():
            s = scores.get(k)
            if s is None:
                continue
            num += w * s
            den += w
        if den == 0:
            return None
        return float(num / den)
    def _embed_one(self, text: str):
        vecs = self.e.embedding_function([text])  # DefaultEmbeddingFunction is callable(texts: List[str]) -> List[List[float]]
        return vecs[0] if vecs else None
    def __init__(self, engine: EngineLike, config: Optional[VerifierConfig] = None):
        self.e: EngineLike = engine
        self.cfg = config or VerifierConfig()
        
        
    # ---------------- public API ----------------

    def _verify_one_reference(
        self,
        extracted_text: str,
        full_text: str,
        ref: Span,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        threshold: float = 0.70,
    ) -> Span:
        """
        Compute per-reference metrics & write a single ensemble MentionVerification
        back into the ReferenceSession (copy), preserving other fields.
        """
        span = self._slice_span(full_text, ref.start_char or 0, ref.end_char or 0)
        # fallbacks: if spans are empty, use entire doc (not ideal, but keeps pipeline moving)
        cited_text = span if span else full_text

        rf = self._score_rapidfuzz(extracted_text, cited_text)
        cv = self._score_coverage(extracted_text, cited_text, min_ngram=min_ngram)
        em = self._score_embedding(extracted_text, cited_text)

        score = self._ensemble({"rapidfuzz": rf, "coverage": cv, "embedding": em}, weights) or 0.0
        is_ok = score >= threshold

        out = ref.model_copy(deep=True)
        # compact, machine-readable notes for reuse later
        detail = {
            "rapidfuzz": rf,
            "coverage": cv,
            "embedding": em,
            "weights": weights,
            "threshold": threshold,
        }
        note = json.dumps(detail, separators=(",", ":"))
        if out.verification is None:
            out.verification = MentionVerification(
                method="ensemble",
                is_verified=is_ok,
                score=score,
                notes=note,
            )
        else:
            out.verification.method = "ensemble"
            out.verification.is_verified = is_ok
            out.verification.score = score
            out.verification.notes = note
        return out
    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: Optional[str] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        update_edges: bool = True,
    ) -> Dict[str, int]:
        """
        Verify all references in nodes (and edges if update_edges=True) for a doc.
        Returns counts of updated items.
        """
        full_text = source_text if source_text is not None else self.e.extract.fetch_document_text(document_id)
        upd_nodes = upd_edges = 0

        # Nodes
        got = self.e.node_collection.get(where={"doc_id": document_id}, include=["documents"])
        for nid, ndoc in zip(got.get("ids") or [], got.get("documents") or []):
            n = Node.model_validate_json(ndoc)
            # what text do we try to validate? prioritize summary, then label
            extracted = n.summary or n.label or ""
            #need-fix
            if not (n.mentions and extracted):
                continue
            new_refs = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                        for r in n.mentions]
            n.mentions = new_refs
            doc, meta = self.e.write.node_doc_and_meta(n)
            self.e.node_collection.update(ids=[nid], documents=[doc], metadatas=[meta])
            self.e.write.index_node_docs(n)
            upd_nodes += 1

        if update_edges:
            got = self.e.edge_collection.get(where={"doc_id": document_id}, include=["documents"])
            for eid, edoc in zip(got.get("ids") or [], got.get("documents") or []):
                e = Edge.model_validate_json(edoc)
                extracted = e.summary or e.label or e.relation or ""
                if not (e.mentions and extracted):
                    continue
                new_refs = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                            for r in e.mentions]
                new_refs = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                            for r in e.mentions]
                e.mentions = new_refs
                doc, meta = self.e.write.edge_doc_and_meta(e)
                self.e.edge_collection.update(ids=[eid], documents=[doc], metadatas=[meta])
                upd_edges += 1

        return {"updated_nodes": upd_nodes, "updated_edges": upd_edges}
    

    def verify_mentions_for_items(
        self,
        items: List[Tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: Optional[Dict[str, str]] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
    ) -> Dict[str, int]:
        """
        Targeted verification for a mixed set of nodes/edges.
        source_text_by_doc lets you pass pre-fetched doc text keyed by doc_id.
        """
        upd_nodes = upd_edges = 0
        for kind, rid in items:
            if kind == "node":
                got = self.e.node_collection.get(ids=[rid], include=["documents", "metadatas"])
                if not got.get("documents"):
                    continue
                n = Node.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (source_text_by_doc or {}).get(doc_id) or self.e.extract.fetch_document_text(doc_id) if doc_id else ""
                extracted = n.summary or n.label or ""
                if not (n.mentions and extracted):
                    continue
                n.mentions = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                                for r in n.mentions]
                doc, meta = self.e.write.node_doc_and_meta(n)
                self.e.node_collection.update(ids=[rid], documents=[doc], metadatas=[meta])
                self.e.write.index_node_docs(n)
                upd_nodes += 1
            elif kind == "edge":
                got = self.e.edge_collection.get(ids=[rid], include=["documents", "metadatas"])
                if not got.get("documents"):
                    continue
                e = Edge.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (source_text_by_doc or {}).get(doc_id) or self.e.extract.fetch_document_text(doc_id) if doc_id else ""
                extracted = e.summary or e.label or e.relation or ""
                if not (e.mentions and extracted):
                    continue
                e.mentions = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                                for r in e.mentions]
                doc, meta = self.e.write.edge_doc_and_meta(e)
                self.e.edge_collection.update(ids=[rid], documents=[doc], metadatas=[meta])
                upd_edges += 1
        to_return = {"updated_nodes": upd_nodes, "updated_edges": upd_edges}
        return to_return
    # # ---------------- internals ----------------
    # def _fetch_doc_text(self, doc_id: str) -> Optional[str]:
    #     got = self.e.document_collection.get(ids=[doc_id], include=["documents"])
    #     docs = got.get("documents") or []
    #     return docs[0] if docs else None

    # def _span_ok(self, ref: ReferenceSession) -> bool:
    #     if ref.end_page < ref.start_page:
    #         return False
    #     if ref.start_page == ref.end_page and ref.end_char < ref.start_char:
    #         return False
    #     return True

    # def _excerpt_from_span(self, text: str, ref: ReferenceSession) -> str:
    #     if not text:
    #         return ""
    #     # naive multi-page handling: treat as linear for now
    #     start = max(0, int(ref.start_char or 0))
    #     end = min(len(text), int(ref.end_char or 0))
    #     if end <= start:
    #         return ""
    #     return text[start:end]

    # def _embed_cosine(self, a: List[float], b: List[float]) -> float:
    #     # tiny cosine helper
    #     dot = sum(x * y for x, y in zip(a, b))
    #     na = math.sqrt(sum(x * x for x in a)) or 1e-8
    #     nb = math.sqrt(sum(y * y for y in b)) or 1e-8
    #     return dot / (na * nb)

    # def _score_ref(self, label_or_summary: str, excerpt: str, obj_embed: Optional[List[float]]) -> Tuple[bool, float, str]:
    #     if not excerpt or len(excerpt) < self.cfg.min_excerpt_len:
    #         return False, 0.0, "excerpt-too-short"

    #     # RapidFuzz score
    #     rf = (fuzz_ratio(label_or_summary, excerpt) or 0.0) / 100.0
    #     notes = [f"rf={rf:.2f}"]

    #     ok = rf >= self.cfg.min_levenshtein_score

    #     # Optional embedding
    #     if ok and self.cfg.use_embeddings and obj_embed and callable(getattr(self.e, "embedding_fn", None)):
    #         try:
    #             s_vec = self.e.embedding_function([excerpt])[0]
    #             cos = self._embed_cosine(obj_embed, s_vec)
    #             notes.append(f"cos={cos:.2f}")
    #             ok = cos >= self.cfg.min_embed_cosine
    #         except Exception:
    #             notes.append("embed-error")

    #     return ok, rf, " | ".join(notes)

    # def _verify_collection_for_doc(self, *, which: str, document_id: str) -> int:
    #     if which == "node":
    #         got = self.e.node_collection.get(where={"doc_id": document_id}, include=["documents", "metadatas"])
    #     else:
    #         got = self.e.edge_collection.get(where={"doc_id": document_id}, include=["documents", "metadatas"])

    #     total_updates = 0
    #     ids = got.get("ids") or []
    #     docs = got.get("documents") or []
    #     metas = got.get("metadatas") or []

    #     for obj_id, doc_json, meta in zip(ids, docs, metas):
    #         obj = Node.model_validate_json(doc_json) if which == "node" else Edge.model_validate_json(doc_json)
    #         changed = False

    #         # pick best text to compare against excerpt
    #         label = getattr(obj, "label", "") or ""
    #         summary = getattr(obj, "summary", "") or ""
    #         text_for_match = summary or label

    #         for ref in (obj.references or []):
    #             did = getattr(ref, "doc_id", None)
    #             if not did and ref.document_page_url:
    #                 # fallback: parse "document/<id>"
    #                 import re
    #                 m = re.search(r"document/([A-Za-z0-9\-]+)", ref.document_page_url)
    #                 if m:
    #                     did = m.group(1)

    #             if did != document_id:
    #                 continue  # only verify mentions for this doc_id

    #             if not self._span_ok(ref):
    #                 ref.verification = MentionVerification(method="span", is_verified=False, score=0.0, notes="bad-span")
    #                 changed = True
    #                 continue

    #             doc_text = self._fetch_doc_text(document_id) or ""
    #             excerpt = ref.excerpt or self._excerpt_from_span(doc_text, ref)

    #             ok, score, notes = self._score_ref(text_for_match, excerpt, getattr(obj, "embedding", None))
    #             ref.verification = MentionVerification(method="heuristic", is_verified=ok, score=score, notes=notes)
    #             changed = True

    #         if changed:
    #             # persist updates
    #             if which == "node":
    #                 self.e.node_collection.update(
    #                     ids=[obj_id],
    #                     documents=[obj.model_dump_json()],
    #                     metadatas=[self.e.chroma_sanitize_metadata({
    #                         "doc_id": (meta or {}).get("doc_id"),
    #                         "label": getattr(obj, "label", None),
    #                         "type": getattr(obj, "type", None),
    #                         "summary": getattr(obj, "summary", None),
    #                         "domain_id": getattr(obj, "domain_id", None),
    #                         "canonical_entity_id": getattr(obj, "canonical_entity_id", None),
    #                         "properties": self.e._json_or_none(getattr(obj, "properties", None)),
    #                         "references": self.e._json_or_none([r.model_dump() for r in (obj.references or [])]),
    #                     })],
    #                 )
    #             else:
    #                 self.e.edge_collection.update(
    #                     ids=[obj_id],
    #                     documents=[obj.model_dump_json()],
    #                     metadatas=[self.e.chroma_sanitize_metadata({
    #                         "doc_id": (meta or {}).get("doc_id"),
    #                         "relation": getattr(obj, "relation", None),
    #                         "source_ids": self.e._json_or_none(getattr(obj, "source_ids", None)),
    #                         "target_ids": self.e._json_or_none(getattr(obj, "target_ids", None)),
    #                         "type": getattr(obj, "type", None),
    #                         "summary": getattr(obj, "summary", None),
    #                         "domain_id": getattr(obj, "domain_id", None),
    #                         "canonical_entity_id": getattr(obj, "canonical_entity_id", None),
    #                         "properties": self.e._json_or_none(getattr(obj, "properties", None)),
    #                         "references": self.e._json_or_none([r.model_dump() for r in (obj.references or [])]),
    #                     })],
    #                 )
    #             total_updates += 1

    #     return total_updates
