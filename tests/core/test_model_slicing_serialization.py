import warnings

import pytest

pytestmark = pytest.mark.ci

from graph_knowledge_engine.engine_core.models import Grounding, Node, Span


def test_model_dump_does_not_warn_for_typed_groundings():
    node = Node(
        label="x",
        type="entity",
        summary="y",
        mentions=[
            Grounding(
                spans=[
                    Span(
                        collection_page_url="test",
                        document_page_url="test",
                        doc_id="doc1",
                        insertion_method="test",
                        page_number=1,
                        start_char=0,
                        end_char=4,
                        excerpt="test",
                        context_before="",
                        context_after="",
                    )
                ]
            )
        ],
        properties={},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = node.model_dump()

    assert dumped["mentions"][0]["spans"][0]["doc_id"] == "doc1"
    assert not [
        w
        for w in caught
        if "PydanticSerializationUnexpectedValue" in str(w.message)
    ]
