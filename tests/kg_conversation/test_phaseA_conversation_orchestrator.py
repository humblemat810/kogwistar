import pytest

from kogwistar.conversation.conversation_orchestrator import (
    ConversationOrchestrator,
    ConversationStepResolver,
)
from kogwistar.conversation.resolvers import default_resolver

pytestmark = [
    pytest.mark.ci,
    pytest.mark.conversation,
    pytest.mark.workflow,
    pytest.mark.ci,
]


def test_phaseA_orchestrator_has_no_inline_step_resolver_factory():
    # Phase A goal: orchestrator should not carry its own per-op resolver factory.
    assert not hasattr(ConversationOrchestrator, "_make_add_turn_step_resolver")


def test_phaseA_conversation_step_resolver_delegates_to_default_resolver():
    r = ConversationStepResolver()

    # Same handler keys (exact match)
    assert set(r.handlers.keys()) == set(default_resolver.handlers.keys())

    # Each handler should be the same underlying function object
    for k, fn in default_resolver.handlers.items():
        assert r.handlers[k] is fn

    # Default fallback should match
    assert r.default is default_resolver.default
