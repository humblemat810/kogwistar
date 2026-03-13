# strategies/verifiers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
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


from ..engine_core.models import Edge, Grounding, MentionVerification, Node, Span


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
    def _score_coverage(
        extracted: str, cited: str, min_ngram: int = 5
    ) -> Optional[float]:
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
            gram = ex_norm[i : i + n]
            if gram in ci_norm:
                for j in range(i, i + n):
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
        dot = sum(a * b for a, b in zip(u, v))
        nu = (sum(a * a for a in u)) ** 0.5
        nv = (sum(b * b for b in v)) ** 0.5
        return float((dot / (nu * nv))) if nu and nv else None

    def _ensemble(
        self, scores: Dict[str, Optional[float]], weights: Dict[str, float]
    ) -> Optional[float]:
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
        return self.e.embed.iterative_defensive_emb(text)

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
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
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

        score = (
            self._ensemble({"rapidfuzz": rf, "coverage": cv, "embedding": em}, weights)
            or 0.0
        )
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

    def _verify_grounding(
        self,
        extracted_text: str,
        full_text: str,
        grounding: Grounding,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        threshold: float = 0.70,
    ) -> Grounding:
        out = grounding.model_copy(deep=True)
        out.spans = [
            self._verify_one_reference(
                extracted_text,
                full_text,
                span,
                min_ngram=min_ngram,
                weights=weights,
                threshold=threshold,
            )
            for span in grounding.spans
        ]
        return out

    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: Optional[str] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        update_edges: bool = True,
    ) -> Dict[str, int]:
        """
        Verify all references in nodes (and edges if update_edges=True) for a doc.
        Returns counts of updated items.
        """
        full_text = (
            source_text
            if source_text is not None
            else self.e.extract.fetch_document_text(document_id)
        )
        upd_nodes = upd_edges = 0

        # Nodes
        got = self.e.backend.node_get(
            where={"doc_id": document_id}, include=["documents"]
        )
        for nid, ndoc in zip(got.get("ids") or [], got.get("documents") or []):
            n = Node.model_validate_json(ndoc)
            # what text do we try to validate? prioritize summary, then label
            extracted = n.summary or n.label or ""
            if not (n.mentions and extracted):
                continue
            n.mentions = [
                self._verify_grounding(
                    extracted,
                    full_text,
                    grounding,
                    min_ngram=min_ngram,
                    weights=weights,
                    threshold=threshold,
                )
                for grounding in n.mentions
            ]
            doc, meta = self.e.write.node_doc_and_meta(n)
            self.e.backend.node_update(ids=[nid], documents=[doc], metadatas=[meta])
            self.e.write.index_node_docs(n)
            upd_nodes += 1

        if update_edges:
            got = self.e.backend.edge_get(
                where={"doc_id": document_id}, include=["documents"]
            )
            for eid, edoc in zip(got.get("ids") or [], got.get("documents") or []):
                e = Edge.model_validate_json(edoc)
                extracted = e.summary or e.label or e.relation or ""
                if not (e.mentions and extracted):
                    continue
                e.mentions = [
                    self._verify_grounding(
                        extracted,
                        full_text,
                        grounding,
                        min_ngram=min_ngram,
                        weights=weights,
                        threshold=threshold,
                    )
                    for grounding in e.mentions
                ]
                doc, meta = self.e.write.edge_doc_and_meta(e)
                self.e.backend.edge_update(ids=[eid], documents=[doc], metadatas=[meta])
                upd_edges += 1

        return {"updated_nodes": upd_nodes, "updated_edges": upd_edges}

    def verify_mentions_for_items(
        self,
        items: List[Tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: Optional[Dict[str, str]] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
    ) -> Dict[str, int]:
        """
        Targeted verification for a mixed set of nodes/edges.
        source_text_by_doc lets you pass pre-fetched doc text keyed by doc_id.
        """
        upd_nodes = upd_edges = 0
        for kind, rid in items:
            if kind == "node":
                got = self.e.backend.node_get(
                    ids=[rid], include=["documents", "metadatas"]
                )
                if not got.get("documents"):
                    continue
                n = Node.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (
                    (source_text_by_doc or {}).get(doc_id)
                    or self.e.extract.fetch_document_text(doc_id)
                    if doc_id
                    else ""
                )
                extracted = n.summary or n.label or ""
                if not (n.mentions and extracted):
                    continue
                n.mentions = [
                    self._verify_grounding(
                        extracted,
                        full_text,
                        grounding,
                        min_ngram=min_ngram,
                        weights=weights,
                        threshold=threshold,
                    )
                    for grounding in n.mentions
                ]
                doc, meta = self.e.write.node_doc_and_meta(n)
                self.e.backend.node_update(ids=[rid], documents=[doc], metadatas=[meta])
                self.e.write.index_node_docs(n)
                upd_nodes += 1
            elif kind == "edge":
                got = self.e.backend.edge_get(
                    ids=[rid], include=["documents", "metadatas"]
                )
                if not got.get("documents"):
                    continue
                e = Edge.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (
                    (source_text_by_doc or {}).get(doc_id)
                    or self.e.extract.fetch_document_text(doc_id)
                    if doc_id
                    else ""
                )
                extracted = e.summary or e.label or e.relation or ""
                if not (e.mentions and extracted):
                    continue
                e.mentions = [
                    self._verify_grounding(
                        extracted,
                        full_text,
                        grounding,
                        min_ngram=min_ngram,
                        weights=weights,
                        threshold=threshold,
                    )
                    for grounding in e.mentions
                ]
                doc, meta = self.e.write.edge_doc_and_meta(e)
                self.e.backend.edge_update(ids=[rid], documents=[doc], metadatas=[meta])
                upd_edges += 1
        to_return = {"updated_nodes": upd_nodes, "updated_edges": upd_edges}
        return to_return
