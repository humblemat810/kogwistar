from __future__ import annotations
from ..models import Node, AdjudicationVerdict, LLMMergeAdjudication, AdjudicationQuestionCode
from langchain.prompts import ChatPromptTemplate

def rule_first_token(left: Node, right: Node) -> AdjudicationVerdict:
    """Toy rule: match on first token of label."""
    lt = (left.label or "").split()[:1]
    rt = (right.label or "").split()[:1]
    same = bool(lt and rt and lt[0].lower() == rt[0].lower())
    return AdjudicationVerdict(same_entity=same, confidence=0.9 if same else 0.2, reason="first-token rule")

def llm_pair(engine, left: Node, right: Node) -> AdjudicationVerdict:
    """Single-pair LLM adjudication using engine.llm."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide if two entities are the SAME real-world entity. Return structured JSON."),
        ("human", "Left:\n{left}\n\nRight:\n{right}")
    ])
    chain = prompt | engine.llm.with_structured_output(LLMMergeAdjudication)
    out: LLMMergeAdjudication = chain.invoke({"left": left.model_dump(), "right": right.model_dump()})
    return out.verdict

def llm_batch(engine, pairs, question_code: AdjudicationQuestionCode = AdjudicationQuestionCode.SAME_ENTITY):
    """Batch adjudication; return list[LLMMergeAdjudication]."""
    mapping = [{"code": int(code)} for code in AdjudicationQuestionCode]
    payload = [{"left": l.model_dump(), "right": r.model_dump(), "question_code": int(question_code)} for l, r in pairs]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You adjudicate candidate pairs. Return only the structured JSON."),
        ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}")
    ])
    # container model approach if your LC version lacks many=True; else swap to many=True.
    from pydantic import BaseModel
    from typing import List
    class Batch(BaseModel):
        items: List[LLMMergeAdjudication]
    chain = prompt | engine.llm.with_structured_output(Batch)
    out: Batch = chain.invoke({"mapping": mapping, "pairs": payload})
    return out.items