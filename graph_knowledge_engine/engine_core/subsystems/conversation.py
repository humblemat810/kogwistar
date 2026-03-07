from __future__ import annotations

from .base import NamespaceProxy


class ConversationSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Service/bootstrap helpers
    def get_conversation_service(self, *args, **kwargs):
        return self._e._get_conversation_service(*args, **kwargs)

    def get_orchestrator(self, *args, **kwargs):
        return self._e._get_orchestrator(*args, **kwargs)

    # Conversation primitive helpers
    def normalize_conversation_edge_metadata(self, *args, **kwargs):
        return self._e._normalize_conversation_edge_metadata(*args, **kwargs)

    def validate_conversation_edge_add(self, *args, **kwargs):
        return self._e._validate_conversation_edge_add(*args, **kwargs)

    def create_conversation_primitive(self, *args, **kwargs):
        return self._e._create_conversation_primitive(*args, **kwargs)

    def get_last_seq_node_internal(self, *args, **kwargs):
        return self._e._get_last_seq_node(*args, **kwargs)

    def get_conversation_tail(self, *args, **kwargs):
        return self._e._get_conversation_tail(*args, **kwargs)

    def where_and(self, *args, **kwargs):
        return self._e._where_and(*args, **kwargs)

    def edge_endpoints_exists(self, *args, **kwargs):
        return self._e._edge_endpoints_exists(*args, **kwargs)

    def edge_endpoints_first_edge_id(self, *args, **kwargs):
        return self._e._edge_endpoints_first_edge_id(*args, **kwargs)

    def conversation_doc_id_for_edge(self, *args, **kwargs):
        return self._e._conversation_doc_id_for_edge(*args, **kwargs)

    def is_duplicate_next_turn_noop(self, *args, **kwargs):
        return self._e._is_duplicate_next_turn_noop(*args, **kwargs)

    # Public conversation API surfaced from engine methods
    def create_conversation(self, *args, **kwargs):
        return self._e.create_conversation(*args, **kwargs)

    def add_conversation_turn(self, *args, **kwargs):
        return self._e.add_conversation_turn(*args, **kwargs)

    def respond_to_utterance(self, *args, **kwargs):
        return self._e.respond_to_utterance(*args, **kwargs)
