from __future__ import annotations

from .base import NamespaceProxy


class ConversationSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "get_conversation_service": "_get_conversation_service",
                "get_orchestrator": "_get_orchestrator",
                "normalize_conversation_edge_metadata": "_normalize_conversation_edge_metadata",
                "validate_conversation_edge_add": "_validate_conversation_edge_add",
                "create_conversation_primitive": "_create_conversation_primitive",
                "get_last_seq_node_internal": "_get_last_seq_node",
                "get_conversation_tail": "_get_conversation_tail",
                "where_and": "_where_and",
                "edge_endpoints_exists": "_edge_endpoints_exists",
                "edge_endpoints_first_edge_id": "_edge_endpoints_first_edge_id",
                "conversation_doc_id_for_edge": "_conversation_doc_id_for_edge",
                "is_duplicate_next_turn_noop": "_is_duplicate_next_turn_noop",
            },
        )

    # Service/bootstrap helpers
    def get_conversation_service(self, *args, **kwargs):
        return self._call("get_conversation_service", *args, **kwargs)

    def get_orchestrator(self, *args, **kwargs):
        return self._call("get_orchestrator", *args, **kwargs)

    # Conversation primitive helpers
    def normalize_conversation_edge_metadata(self, *args, **kwargs):
        return self._call("normalize_conversation_edge_metadata", *args, **kwargs)

    def validate_conversation_edge_add(self, *args, **kwargs):
        return self._call("validate_conversation_edge_add", *args, **kwargs)

    def create_conversation_primitive(self, *args, **kwargs):
        return self._call("create_conversation_primitive", *args, **kwargs)

    def get_last_seq_node_internal(self, *args, **kwargs):
        return self._call("get_last_seq_node_internal", *args, **kwargs)

    def get_conversation_tail(self, *args, **kwargs):
        return self._call("get_conversation_tail", *args, **kwargs)

    def where_and(self, *args, **kwargs):
        return self._call("where_and", *args, **kwargs)

    def edge_endpoints_exists(self, *args, **kwargs):
        return self._call("edge_endpoints_exists", *args, **kwargs)

    def edge_endpoints_first_edge_id(self, *args, **kwargs):
        return self._call("edge_endpoints_first_edge_id", *args, **kwargs)

    def conversation_doc_id_for_edge(self, *args, **kwargs):
        return self._call("conversation_doc_id_for_edge", *args, **kwargs)

    def is_duplicate_next_turn_noop(self, *args, **kwargs):
        return self._call("is_duplicate_next_turn_noop", *args, **kwargs)

    # Public conversation API surfaced from engine methods
    def create_conversation(self, *args, **kwargs):
        return self._call("create_conversation", *args, **kwargs)

    def add_conversation_turn(self, *args, **kwargs):
        return self._call("add_conversation_turn", *args, **kwargs)

    def respond_to_utterance(self, *args, **kwargs):
        return self._call("respond_to_utterance", *args, **kwargs)
