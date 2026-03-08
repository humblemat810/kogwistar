# graph_knowledge_engine/strategies/adjudicators.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple

from joblib import Memory
from pydantic import BaseModel

from .types import EngineLike, IAdjudicator
from ..engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationTarget,
    BatchAdjudications,
    Edge,
    LLMMergeAdjudication,
    Node,
    QUESTION_DESC,
    QUESTION_KEY,
)
from ..llm_tasks import AdjudicateBatchTaskRequest, AdjudicatePairTaskRequest


@dataclass(frozen=True)
class PairAdjudicationTrace:
    adjudication: LLMMergeAdjudication | None
    raw: object | None
    parsing_error: str | None


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _cacheable_raw(raw: object | None) -> object | None:
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, (list, dict)):
        return raw
    return repr(raw)


def _invoke_adjudicate_pair_task(
    pair_task,
    *,
    question: str,
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    result = pair_task(
        AdjudicatePairTaskRequest(
            question=question,
            left=left,
            right=right,
        )
    )
    verdict_payload = result.verdict_payload
    return {
        "verdict_payload": dict(verdict_payload) if verdict_payload is not None else None,
        "raw": _cacheable_raw(result.raw),
        "parsing_error": result.parsing_error,
    }


class LLMPairAdjudicatorImpl:
    """Inline pair adjudicator using typed llm task contracts."""

    def __init__(self, engine: EngineLike):
        self.e = engine

    def adjudicate_pair(
        self,
        left: Any,
        right: Any,
        *,
        question: str = "same_entity",
    ) -> LLMMergeAdjudication:
        payload_left = getattr(left, "model_dump", lambda: left)()
        payload_right = getattr(right, "model_dump", lambda: right)()
        result = self.e.llm_tasks.adjudicate_pair(
            AdjudicatePairTaskRequest(
                question=question,
                left=payload_left if isinstance(payload_left, dict) else {"value": payload_left},
                right=payload_right if isinstance(payload_right, dict) else {"value": payload_right},
            )
        )
        if result.verdict_payload is None:
            raise ValueError(f"Pair adjudication failed: {result.parsing_error or 'missing verdict payload'}")
        return LLMMergeAdjudication.model_validate(result.verdict_payload)


class LLMBatchAdjudicatorImpl:
    """Inline batch adjudicator (cross-kind aware)."""

    def __init__(self, engine: EngineLike):
        self.e = engine

    @staticmethod
    def _kind(x: Any) -> str:
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
        attrs: Dict[str, Any] = {}
        for k in ("relation", "date", "country", "ticker", "role"):
            if d.get(k) is not None:
                attrs[k] = d[k]
        if attrs:
            out["attrs"] = attrs
        props = d.get("properties") or {}
        sig = props.get("signature_text")
        if sig:
            out["signature"] = str(sig)
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

        uniq: List[Tuple[str, str, Any]] = []
        seen = set()
        for l, r in pairs:
            for x in (l, r):
                k = self._kind(x)
                i = self._id(x)
                t = (k, i)
                if t in seen:
                    continue
                seen.add(t)
                uniq.append((k, i, x))

        alias_for: Dict[Tuple[str, str], str] = {(k, i): f"n{idx}" for idx, (k, i, _) in enumerate(uniq)}
        inv_alias: Dict[str, Tuple[str, str]] = {v: k for k, v in alias_for.items()}

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

        pair_payload: list[dict[str, Any]] = []
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

        result = self.e.llm_tasks.adjudicate_batch(
            AdjudicateBatchTaskRequest(mapping=mapping_table, pairs=pair_payload)
        )
        out_items = [LLMMergeAdjudication.model_validate(item) for item in result.verdict_payloads]

        result_items: List[LLMMergeAdjudication] = []
        for item in out_items:
            md = item.model_dump()
            la = md.get("left_id")
            ra = md.get("right_id")
            if la and la in inv_alias:
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

    def adjudicate_pair_trace(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        question: str,
        *,
        cache_dir: str | PathLike[str] | None = None,
    ) -> PairAdjudicationTrace:
        if (left.kind != right.kind) and question != "node_edge_equivalence":
            raise ValueError("Cross-kind only allowed for 'node_edge_equivalence'")
        if not self.e.allow_cross_kind_adjudication and question == "node_edge_equivalence":
            raise ValueError("Cross-kind adjudication disabled")

        left_payload = _normalize_payload(left.model_dump())
        right_payload = _normalize_payload(right.model_dump())
        pair_task = self.e.llm_tasks.adjudicate_pair

        if cache_dir is not None:
            memory = Memory(location=str(cache_dir), verbose=0)
            payload = memory.cache(_invoke_adjudicate_pair_task, ignore=["pair_task"])(
                pair_task=pair_task,
                question=question,
                left=left_payload,
                right=right_payload,
            )
        else:
            payload = _invoke_adjudicate_pair_task(
                pair_task,
                question=question,
                left=left_payload,
                right=right_payload,
            )

        verdict_payload = payload.get("verdict_payload")
        adjudication: Optional[LLMMergeAdjudication]
        parsing_error = payload.get("parsing_error")
        if verdict_payload is None:
            adjudication = None
        else:
            try:
                adjudication = LLMMergeAdjudication.model_validate(verdict_payload)
            except Exception as exc:
                adjudication = None
                parsing_error = f"{parsing_error or 'invalid verdict payload'} | validation_error: {exc}"
        return PairAdjudicationTrace(
            adjudication=adjudication,
            raw=payload.get("raw"),
            parsing_error=parsing_error,
        )

    def adjudicate_pair(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        question: str,
    ) -> Dict[Any, Any] | BaseModel:
        trace = self.adjudicate_pair_trace(left, right, question)
        if trace.adjudication is None:
            raise ValueError(f"Pair adjudication failed: {trace.parsing_error or 'missing verdict payload'}")
        return trace.adjudication

    def adjudicate_merge(self, left_node: Node | Edge, right_node: Node | Edge) -> Dict[Any, Any] | BaseModel:
        left = self.e.adjudicate.target_from_node(left_node) if isinstance(left_node, Node) else self.e.adjudicate.target_from_edge(left_node)
        right = self.e.adjudicate.target_from_node(right_node) if isinstance(right_node, Node) else self.e.adjudicate.target_from_edge(right_node)
        question = "same_entity" if left.kind == "node" else "same_relation"
        return self.adjudicate_pair(left, right, question=question)

    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ):
        if not pairs:
            return []

        qcode = AdjudicationQuestionCode(question_code)
        qkey = QUESTION_KEY[qcode]

        def node_id(n: Node):
            return getattr(n, "id", None) or n.model_dump().get("id")

        def node_kind(n: Node):
            return (
                getattr(n, "kind", None)
                or n.model_dump().get("kind")
                or getattr(n, "type", None)
                or n.model_dump().get("type")
                or n.__class__.__name__
            )

        def normalized_signature(left, right):
            lid, rid = node_id(left), node_id(right)
            lkind, rkind = node_kind(left), node_kind(right)
            a, b = (lkind, str(lid)), (rkind, str(rid))
            key_pair = (a, b) if a <= b else (b, a)
            return key_pair + (qkey,)

        def compact_payload(n: Node):
            d = n.model_dump()
            out: Dict[str, Any] = {}
            out["kind"] = node_kind(n)
            out["type"] = d.get("type")
            out["name"] = d.get("name") or d.get("label") or d.get("title")
            attrs: Dict[str, Any] = {}
            for k in ("dob", "country", "ticker", "date", "role", "source"):
                if k in d and d[k] is not None:
                    attrs[k] = d[k]
            if attrs:
                out["attrs"] = attrs
            if "signature" in d and d["signature"]:
                out["signature"] = d["signature"]
            return out

        cache: dict[tuple[Any, ...], LLMMergeAdjudication] = {}
        known_by_index: dict[int, LLMMergeAdjudication] = {}
        unknown_indices: list[int] = []
        unknown_pairs: list[tuple[Node, Node, tuple[Any, ...]]] = []

        for idx, (left, right) in enumerate(pairs):
            k = normalized_signature(left, right)
            if k in cache:
                known_by_index[idx] = cache[k]
                continue
            unknown_indices.append(idx)
            unknown_pairs.append((left, right, k))

        if not unknown_pairs:
            ordered = [known_by_index[i] for i in range(len(pairs))]
            return ordered, qkey

        uniq_objs: list[tuple[str, str]] = []
        seen = set()
        for left, right, _ in unknown_pairs:
            for n in (left, right):
                tup = (node_kind(n), str(node_id(n)))
                if tup in seen:
                    continue
                seen.add(tup)
                uniq_objs.append(tup)

        alias_map = {obj: f"n{i}" for i, obj in enumerate(uniq_objs)}
        inv_alias = {v: obj for obj, v in alias_map.items()}

        mapping_table = [
            {"code": int(code), "key": QUESTION_KEY[code], "description": QUESTION_DESC[code]}
            for code in AdjudicationQuestionCode
        ]

        def _fmt(ctx):
            meta = f"[doc={ctx['doc_id']} p{ctx.get('start_page')}-{ctx.get('end_page')}]" if ctx.get("doc_id") else ""
            return f"{meta} ...{(ctx['context'] or ctx['mention'] or '')}..."

        adjudication_inputs: list[dict[str, Any]] = []
        for left, right, _ in unknown_pairs:
            l_key = (node_kind(left), str(node_id(left)))
            r_key = (node_kind(right), str(node_id(right)))
            left_ctxs = self.e.read.extract_reference_contexts(left, window_chars=260, max_contexts=2)
            right_ctxs = self.e.read.extract_reference_contexts(right, window_chars=260, max_contexts=2)
            left_blurbs = "\n".join(_fmt(c) for c in left_ctxs)
            right_blurbs = "\n".join(_fmt(c) for c in right_ctxs)
            adjudication_inputs.append(
                {
                    "left": {"id": alias_map[l_key], **compact_payload(left)},
                    "left_context": left_blurbs,
                    "right": {"id": alias_map[r_key], **compact_payload(right)},
                    "right_context": right_blurbs,
                    "cross_type": node_kind(left) != node_kind(right),
                    "question_code": int(qcode),
                }
            )

        result = self.e.llm_tasks.adjudicate_batch(
            AdjudicateBatchTaskRequest(mapping=mapping_table, pairs=adjudication_inputs)
        )
        parsed_items = [LLMMergeAdjudication.model_validate(item) for item in result.verdict_payloads]

        fixed_results: list[LLMMergeAdjudication] = []
        for (left, right, key_sig), res in zip(unknown_pairs, parsed_items):
            left_alias = getattr(res, "left_id", None) or res.model_dump().get("left_id")
            right_alias = getattr(res, "right_id", None) or res.model_dump().get("right_id")
            l_kind, l_id = inv_alias.get(left_alias, (node_kind(left), str(node_id(left))))
            r_kind, r_id = inv_alias.get(right_alias, (node_kind(right), str(node_id(right))))

            if hasattr(res, "model_copy"):
                res = res.model_copy(update={"left_id": l_id, "right_id": r_id, "left_kind": l_kind, "right_kind": r_kind})
            else:
                setattr(res, "left_id", l_id)
                setattr(res, "right_id", r_id)
                setattr(res, "left_kind", l_kind)
                setattr(res, "right_kind", r_kind)

            cache[key_sig] = res
            fixed_results.append(res)

        ordered = [None] * len(pairs)
        for i, r in known_by_index.items():
            ordered[i] = r
        it = iter(fixed_results)
        for i in unknown_indices:
            ordered[i] = next(it)

        return ordered, qkey
