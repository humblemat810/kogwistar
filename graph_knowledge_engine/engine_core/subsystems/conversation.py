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
