# strategies/merge_policies.py
from __future__ import annotations
import uuid
from ..models import Node, Edge, AdjudicationVerdict, ReferenceSession, MentionVerification

def prefer_existing_canonical(engine, left: Node, right: Node, verdict: AdjudicationVerdict) -> str:
    """Your current default policy, factored out."""
    if not verdict.same_entity:
        raise ValueError("Verdict not positive.")
    cid = verdict.canonical_entity_id or (left.canonical_entity_id or right.canonical_entity_id) or str(uuid.uuid1())
    # set & persist
    left.canonical_entity_id = cid
    right.canonical_entity_id = cid
    # persist nodes
    for n in (left, right):
        doc, meta = engine._node_doc_and_meta(n)  # using your helper
        engine.node_collection.update(ids=[n.id], documents=[doc], metadatas=[meta])
    # create same_as edge with evidence from both
    def _best_ref(n: Node) -> ReferenceSession:
        if n.references:
            r = sorted(n.references, key=lambda x: (getattr(x, "start_page", 9e9), getattr(x, "start_char", 9e9)))[0].model_copy(deep=True)
            r.verification = r.verification or MentionVerification(method="heuristic", is_verified=True, score=0.5, notes="adjudication evidence")
            return r
        return ReferenceSession(collection_page_url=f"document_collection/{n.doc_id}", document_page_url=f"document/{n.doc_id}", start_page=1, end_page=1, start_char=0, end_char=0)
    edge = Edge(
        label="same_as", type="relationship", summary=verdict.reason or "merge",
        relation="same_as", source_ids=[left.id], target_ids=[right.id],
        properties={"confidence": verdict.confidence}, references=[_best_ref(left), _best_ref(right)]
    )
    engine.add_edge(edge, doc_id="__adjudication__")
    return cid

def transitive_close_same_as(engine, seed_ids: list[str]) -> None:
    """Optional: after merges, compute connected components of same_as and normalize star anchors."""
    # You can call your existing rebalance helper here if you like.
    pass