# graph_knowledge_engine/strategies/adjudicators.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate

from graph_knowledge_engine.strategies import EngineLike

from ..models import (
    LLMMergeAdjudication,
    AdjudicationQuestionCode,
    QUESTION_KEY,
    QUESTION_DESC,
)

# Container to avoid List[T] issues with with_structured_output
class BatchAdjudications(BaseModel):
    items: List[LLMMergeAdjudication]

class LLMPairAdjudicatorImpl:
    """
    Inline pair adjudicator: builds its own prompt and calls engine.llm directly.
    No callbacks into engine logic.
    """

    def __init__(self, engine: EngineLike):
        # Needs access to engine.llm only (and maybe engine-side config later)
        self.e = engine

    def adjudicate_pair(
        self,
        left: Any,
        right: Any,
        *,
        question: str = "same_entity",
    ) -> LLMMergeAdjudication:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You adjudicate whether two objects refer to the SAME real-world thing. "
                    "Be conservative and return a structured JSON verdict only."
                ),
                (
                    "human",
                    "Question: {question}\n\n"
                    "Left:\n{left}\n\nRight:\n{right}\n\n"
                    "Return only the structured JSON for the schema."
                ),
            ]
        )
        # Keep payload small: rely on model_dump() from Pydantic objects
        payload_left = getattr(left, "model_dump", lambda: left)()
        payload_right = getattr(right, "model_dump", lambda: right)()

        chain = prompt | self.e.llm.with_structured_output(LLMMergeAdjudication)
        return chain.invoke({"question": question, "left": payload_left, "right": payload_right})


class LLMBatchAdjudicatorImpl:
    """
    Inline batch adjudicator (cross-kind aware).
    - Aliases objects to short ids to shrink prompts.
    - Accepts ANY pair types (node↔node, edge↔edge, cross‑type).
    - Returns (list[LLMMergeAdjudication], qkey).
    """

    def __init__(self, engine: EngineLike):
        self.e = engine

    # -------- helpers --------
    @staticmethod
    def _kind(x: Any) -> str:
        # prefer 'type' if present, else class
        return getattr(x, "type", None) or getattr(x, "__class__", type("X", (), {})).__name__

    @staticmethod
    def _id(x: Any) -> str:
        return getattr(x, "id", None) or getattr(x, "model_dump", lambda: {})().get("id") or ""

    @staticmethod
    def _compact(x: Any) -> Dict[str, Any]:
        d = getattr(x, "model_dump", lambda: x)()
        if not isinstance(d, dict):
            return {"kind": LLMBatchAdjudicatorImpl._kind(x)}
        out: Dict[str, Any] = {
            "kind": d.get("type") or LLMBatchAdjudicatorImpl._kind(x),
            "type": d.get("type"),
            "name": d.get("label") or d.get("name") or d.get("title"),
            "summary": d.get("summary"),
        }
        # small whitelist of attributes frequently useful
        attrs = {}
        for k in ("relation", "date", "country", "ticker", "role"):
            if d.get(k) is not None:
                attrs[k] = d[k]
        if attrs:
            out["attrs"] = attrs
        props = d.get("properties") or {}
        sig = props.get("signature_text")
        if sig:
            out["signature"] = str(sig)
        # signature of hyperedge endpoints if present (kept short)
        for field in ("source_ids", "target_ids", "source_edge_ids", "target_edge_ids"):
            if d.get(field):
                out[field] = d[field]
        return out

    def adjudicate_batch(
        self,
        pairs: List[Tuple[Any, Any]],
        *,
        question_code: int,
    ) -> Tuple[List[LLMMergeAdjudication], str]:

        if not pairs:
            return [], QUESTION_KEY[AdjudicationQuestionCode(question_code)]

        qcode = AdjudicationQuestionCode(question_code)
        qkey = QUESTION_KEY[qcode]

        # 1) Collect unique objects to alias
        uniq: List[Tuple[str, str, Any]] = []  # (kind, id, obj)
        seen = set()
        for l, r in pairs:
            for x in (l, r):
                k = self._kind(x)
                i = self._id(x)
                t = (k, i)
                if t not in seen:
                    seen.add(t)
                    uniq.append((k, i, x))

        alias_for: Dict[Tuple[str, str], str] = {(k, i): f"n{idx}" for idx, (k, i, _) in enumerate(uniq)}
        inv_alias: Dict[str, Tuple[str, str]] = {v: k for k, v in alias_for.items()}

        # 2) Build compact items with aliased ids
        mapping_table = [
            {"code": int(code), "key": QUESTION_KEY[code], "description": QUESTION_DESC[code]}
            for code in AdjudicationQuestionCode
        ]

        compact: Dict[str, Dict[str, Any]] = {}
        for k, i, x in uniq:
            aid = alias_for[(k, i)]
            item = self._compact(x)
            item["id"] = aid
            compact[aid] = item

        # 3) Build pair list for LLM
        pair_payload = []
        for l, r in pairs:
            la = alias_for[(self._kind(l), self._id(l))]
            ra = alias_for[(self._kind(r), self._id(r))]
            pair_payload.append(
                {
                    "left": compact[la],
                    "right": compact[ra],
                    "question_code": int(qcode),
                    "cross_type": compact[la]["kind"] != compact[ra]["kind"],
                }
            )

        # 4) Prompt & call
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You adjudicate candidate pairs (entity↔entity, relation↔relation, and cross-type). "
                    "Use the mapping table to interpret question_code. "
                    "Return only the structured JSON per schema. Use the short ids exactly.",
                ),
                (
                    "human",
                    "Mapping table:\n{mapping}\n\nPairs:\n{pairs}",
                ),
            ]
        )
        chain = prompt | self.e.llm.with_structured_output(BatchAdjudications)
        out: BatchAdjudications = chain.invoke({"mapping": mapping_table, "pairs": pair_payload})

        # 5) De-alias ids (left_id/right_id) if present in model
        result_items: List[LLMMergeAdjudication] = []
        for item in out.items:
            md = item.model_dump()
            la = md.get("left_id")
            ra = md.get("right_id")
            if la and la in inv_alias:
                # replace alias with original id string
                _, real_l = inv_alias[la]
                md["left_id"] = real_l
            if ra and ra in inv_alias:
                _, real_r = inv_alias[ra]
                md["right_id"] = real_r
            result_items.append(LLMMergeAdjudication.model_validate(md))

        return result_items, qkey
