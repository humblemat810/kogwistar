from __future__ import annotations

from kogwistar.conversation.models import EvidencePackDigest as ConversationEvidencePackDigest
from kogwistar.provenance import EvidencePackDigest, evidence_pack_digest_hash


def test_evidence_pack_digest_hash_is_stable_for_reordered_ids():
    a = EvidencePackDigest(
        node_ids=["node-b", "node-a"],
        edge_ids=["edge-2", "edge-1"],
        depth="deep",
        max_chars_per_item=128,
        max_total_chars=512,
        source="alpha",
    )
    b = EvidencePackDigest(
        node_ids=["node-a", "node-b"],
        edge_ids=["edge-1", "edge-2"],
        depth="deep",
        max_chars_per_item=128,
        max_total_chars=512,
        source="alpha",
    )
    a.evidence_pack_hash = "stored-hash"
    b.evidence_pack_hash = "different-stored-hash"

    assert evidence_pack_digest_hash(a) == evidence_pack_digest_hash(b)
    assert a.model_dump()["source"] == "alpha"


def test_conversation_evidence_pack_digest_is_the_shared_core_type():
    assert ConversationEvidencePackDigest is EvidencePackDigest
