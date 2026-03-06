# graph_knowledge_engine/strategies/adjudicators.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .types import EngineLike, IAdjudicator

from ..engine_core.models import (
    LLMMergeAdjudication,
    AdjudicationQuestionCode,
    AdjudicationTarget,
    BatchAdjudications,
    Node, Edge,
    QUESTION_KEY,
    QUESTION_DESC,
)


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

class Adjudicator(IAdjudicator):
    def __init__(self, engine: EngineLike):
        self.e = engine
    def adjudicate_pair(self, left: AdjudicationTarget, right: AdjudicationTarget, question: str)-> Dict[Any, Any] | BaseModel:
        if (left.kind != right.kind) and question != "node_edge_equivalence":
            raise ValueError("Cross-kind only allowed for 'node_edge_equivalence'")
        if not self.e.allow_cross_kind_adjudication and question == "node_edge_equivalence":
            raise ValueError("Cross-kind adjudication disabled")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a careful adjudicator. Decide if LEFT and RIGHT correspond as the question specifies.\n"
            "- If question == 'same_entity': same real-world entity (node↔node).\n"
            "- If question == 'same_relation': same logical relation instance (edge↔edge) including endpoints.\n"
            "- If question == 'node_edge_equivalence': determine if the NODE is a named reification/denotation "
            "  of the EDGE (i.e., the node represents that specific relation instance). "
            "Be conservative; return JSON verdict."),
            ("human", "Question: {question}\nLeft:\n{left}\nRight:\n{right}")
        ])
        chain = prompt | self.e.llm.with_structured_output(LLMMergeAdjudication)
        return chain.invoke({"question": question,
                            "left": left.model_dump(),
                            "right": right.model_dump()})
    def adjudicate_merge(self, left_node: Node | Edge, right_node: Node | Edge) -> Dict[Any, Any] | BaseModel:
        # Back-compat wrapper if you still call with concrete models
        left = self.e._target_from_node(left_node) if isinstance(left_node, Node) else self.e._target_from_edge(left_node)
        right = self.e._target_from_node(right_node) if isinstance(right_node, Node) else self.e._target_from_edge(right_node)
        question = "same_entity" if left.kind == "node" else "same_relation"
        return self.adjudicate_pair(left, right, question=question)
    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ):# -> list[Any] | tuple[list[Any], str] | tuple[list[None], str]:
        if not pairs:
            return []

        qcode = AdjudicationQuestionCode(question_code)
        qkey = QUESTION_KEY[qcode]

        # --- helpers -------------------------------------------------------------
        def node_id(n: Node):
            return getattr(n, "id", None) or n.model_dump().get("id")

        def node_kind(n: Node):
            # prefer an explicit 'kind' (e.g., 'entity'|'relation') then fallback to type/classname
            return (
                getattr(n, "kind", None)
                or n.model_dump().get("kind")
                or getattr(n, "type", None)
                or n.model_dump().get("type")
                or n.__class__.__name__
            )

        def normalized_signature(left, right):
            """Normalized cache key for cross-type pairs, including qkey so different questions don't collide."""
            lid, rid = node_id(left), node_id(right)
            lkind, rkind = node_kind(left), node_kind(right)
            a, b = (lkind, str(lid)), (rkind, str(rid))
            key_pair = (a, b) if a <= b else (b, a)
            return key_pair + (qkey,)  # ((kind,id),(kind,id), qkey)

        # compact copy of node for LLM (keep it tiny but informative)
        def compact_payload(n: Node):
            d = n.model_dump()
            out = {}
            # always include these:
            out["kind"] = node_kind(n)
            out["type"] = d.get("type")  # coarse class if present
            out["name"] = d.get("name") or d.get("label") or d.get("title")
            # optional small attrs whitelist if present
            attrs = {}
            for k in ("dob", "country", "ticker", "date", "role", "source"):
                if k in d and d[k] is not None:
                    attrs[k] = d[k]
            if attrs:
                out["attrs"] = attrs
            # relation/hyperedge signature if present
            if "signature" in d and d["signature"]:
                # expect [{"role": "...", "id": "...", ...}, ...]
                out["signature"] = d["signature"]
            return out

        # --- 1) pre-pass: cache lookup & collect unknowns -----------------------
        cache = {}  # key: ((kind,id),(kind,id), qkey) -> LLMMergeAdjudication
        known_by_index = {}
        unknown_indices = []
        unknown_pairs = []

        for idx, (left, right) in enumerate(pairs):
            k = normalized_signature(left, right)
            if k in cache:
                known_by_index[idx] = cache[k]
            else:
                unknown_indices.append(idx)
                unknown_pairs.append((left, right, k))

        if not unknown_pairs:
            ordered = [known_by_index[i] for i in range(len(pairs))]
            return ordered, qkey

        # --- 2) short-id aliasing over unique (kind,id) objects -----------------
        # collect unique objects by (kind,id)
        uniq_objs = []
        seen = set()
        for left, right, _ in unknown_pairs:
            for n in (left, right):
                tup = (node_kind(n), str(node_id(n)))
                if tup not in seen:
                    seen.add(tup)
                    uniq_objs.append(tup)

        alias_map = {obj: f"n{i}" for i, obj in enumerate(uniq_objs)}  # (kind,id) -> "nX"
        inv_alias = {v: obj for obj, v in alias_map.items()}           # "nX" -> (kind,id)

        # --- 3) build tiny LLM inputs (aliased ids + compact fields) ------------
        mapping_table = [
            {"code": int(code), "key": QUESTION_KEY[code], "description": QUESTION_DESC[code]}
            for code in AdjudicationQuestionCode
        ]
        def _fmt(ctx):
            meta = f"[doc={ctx['doc_id']} p{ctx.get('start_page')}–{ctx.get('end_page')}]" if ctx.get('doc_id') else ""
            return f"{meta} …{(ctx['context'] or ctx['mention'] or '')}…"

        
        adjudication_inputs = []
        for left, right, _ in unknown_pairs:
            l_key = (node_kind(left), str(node_id(left)))
            r_key = (node_kind(right), str(node_id(right)))
            left_ctxs  = self.e.extract_reference_contexts(left,  window_chars=260, max_contexts=2)
            right_ctxs = self.e.extract_reference_contexts(right, window_chars=260, max_contexts=2)
            left_blurbs  = "\n".join(_fmt(c) for c in left_ctxs)
            right_blurbs = "\n".join(_fmt(c) for c in right_ctxs)
            adjudication_inputs.append({
                "left":  {"id": alias_map[l_key],  **compact_payload(left)},
                "left_context": left_blurbs,
                "right": {"id": alias_map[r_key], **compact_payload(right)},
                "right_context": right_blurbs,
                "cross_type": node_kind(left) != node_kind(right),
                "question_code": int(qcode),
            })

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You adjudicate candidate pairs, including cross-type pairs (entity↔relation, etc.). "
            "Use the mapping table to interpret question_code. "
            "This is for hypergraph application, equalize if a node is representing a relationship indicated by an edge. "
            "Return only the structured JSON per schema. Use the short ids exactly."),
            ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}")
        ])
        
        chain = prompt | self.e.llm.with_structured_output(BatchAdjudications, include_raw= True)
        llm_results= chain.invoke({"mapping": mapping_table, "pairs": adjudication_inputs})
        raw = llm_results.get("raw") if isinstance(llm_results, dict) else None
        parsed: BatchAdjudications  = llm_results.get("parsed") if isinstance(llm_results, dict) else llm_results
        err = llm_results.get("parsing_error") if isinstance(llm_results, dict) else None
        # --- 4) de-alias result ids and cache by normalized key -----------------
        fixed_results = []
        for (left, right, key_sig), res in zip(unknown_pairs, parsed.merge_adjudications):
            # map "nX" back to (kind,id)
            left_alias = getattr(res, "left_id", None) or res.model_dump().get("left_id")
            right_alias = getattr(res, "right_id", None) or res.model_dump().get("right_id")
            l_kind, l_id = inv_alias.get(left_alias, (node_kind(left), str(node_id(left))))
            r_kind, r_id = inv_alias.get(right_alias, (node_kind(right), str(node_id(right))))

            # rebuild or mutate to set original ids
            if hasattr(res, "model_copy"):
                res = res.model_copy(update={"left_id": l_id, "right_id": r_id, "left_kind": l_kind, "right_kind": r_kind})
            else:
                setattr(res, "left_id", l_id); setattr(res, "right_id", r_id)
                setattr(res, "left_kind", l_kind); setattr(res, "right_kind", r_kind)

            cache[key_sig] = res
            fixed_results.append(res)

        # --- 5) stitch results back to original order ---------------------------
        ordered = [None] * len(pairs)
        for i, r in known_by_index.items():
            ordered[i] = r
        it = iter(fixed_results)
        for i in unknown_indices:
            ordered[i] = next(it)

        return ordered, qkey