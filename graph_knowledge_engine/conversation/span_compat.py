from __future__ import annotations

from graph_knowledge_engine.engine_core.models import MentionVerification
from graph_knowledge_engine.engine_core.models import Span


def from_dummy_for_conversation(doc_id: str = "_conv:_dummy"):
    if doc_id.startswith("_conv:"):
        pass
    else:
        doc_id = "_conv:" + doc_id
    return Span(
        collection_page_url=f"conversation/{doc_id}",
        document_page_url=f"conversation/{doc_id}",
        doc_id=f"{doc_id}",
        insertion_method="system",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes=""
        ),
    )


def install_span_compat_aliases() -> None:
    # Keep legacy name in the chat module boundary, not in engine_core.
    setattr(
        Span, "from_dummy_for_conversation", staticmethod(from_dummy_for_conversation)
    )
