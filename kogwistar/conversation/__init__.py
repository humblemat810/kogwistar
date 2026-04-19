"""Conversation-layer compatibility entrypoints.
TO-DO: Move the real file to the location if they are in other modules
Use lazy imports to avoid package-init circular imports during migration.
"""

from typing import TYPE_CHECKING
from kogwistar.conversation.span_compat import install_span_compat_aliases

install_span_compat_aliases()


if TYPE_CHECKING:
    from kogwistar.conversation.designer import (
        ConversationWorkflowDesigner,
        AgenticAnsweringWorkflowDesigner,
    )
    from kogwistar.conversation.agentic_answering import (
        AgenticAnsweringAgent,
    )
    from kogwistar.conversation.conversation_context import (
        ConversationContextBuilder,
        ContextSources,
        PromptContext,
    )
    from kogwistar.conversation.conversation_orchestrator import (
        ConversationOrchestrator,
    )
    from kogwistar.conversation.conversation_state_contracts import (
        ConversationWorkflowState,
    )
    from kogwistar.conversation.knowledge_retriever import (
        KnowledgeRetriever,
    )
    from kogwistar.conversation.memory_retriever import MemoryRetriever
    from kogwistar.conversation.retrieval_orchestrator import (
        RetrievalOrchestrator,
    )
    from kogwistar.conversation.service import ConversationService
    from kogwistar.conversation.tool_runner import ToolRunner
    from kogwistar.conversation.tool_registry import (
        ToolDefinition,
        ToolReceipt,
        ToolRegistry,
        ToolRequirement,
    )

__all__ = [
    "AgenticAnsweringAgent",
    "ConversationWorkflowDesigner",
    "AgenticAnsweringWorkflowDesigner",
    "ConversationContextBuilder",
    "ContextSources",
    "PromptContext",
    "ConversationOrchestrator",
    "ConversationWorkflowState",
    "KnowledgeRetriever",
    "MemoryRetriever",
    "RetrievalOrchestrator",
    "ConversationService",
    "ToolRunner",
    "ToolDefinition",
    "ToolReceipt",
    "ToolRegistry",
    "ToolRequirement",
]


def __getattr__(name: str):
    if name == "AgenticAnsweringAgent":
        from kogwistar.conversation.agentic_answering import (
            AgenticAnsweringAgent,
        )

        return AgenticAnsweringAgent
    if name in ("ConversationContextBuilder", "ContextSources", "PromptContext"):
        from kogwistar.conversation.conversation_context import (
            ConversationContextBuilder,
            ContextSources,
            PromptContext,
        )

        return {
            "ConversationContextBuilder": ConversationContextBuilder,
            "ContextSources": ContextSources,
            "PromptContext": PromptContext,
        }[name]
    if name == "ConversationOrchestrator":
        from kogwistar.conversation.conversation_orchestrator import (
            ConversationOrchestrator,
        )

        return ConversationOrchestrator
    if name == "WorkflowState":
        from kogwistar.conversation.conversation_state_contracts import (
            ConversationWorkflowState,
        )

        return ConversationWorkflowState
    if name == "KnowledgeRetriever":
        from kogwistar.conversation.knowledge_retriever import (
            KnowledgeRetriever,
        )

        return KnowledgeRetriever
    if name == "MemoryRetriever":
        from kogwistar.conversation.memory_retriever import MemoryRetriever

        return MemoryRetriever
    if name == "RetrievalOrchestrator":
        from kogwistar.conversation.retrieval_orchestrator import (
            RetrievalOrchestrator,
        )

        return RetrievalOrchestrator
    if name == "ConversationService":
        from kogwistar.conversation.service import ConversationService

        return ConversationService
    if name == "ToolRunner":
        from kogwistar.conversation.tool_runner import ToolRunner

        return ToolRunner
    if name in ("ToolDefinition", "ToolReceipt", "ToolRegistry", "ToolRequirement"):
        from kogwistar.conversation.tool_registry import (
            ToolDefinition,
            ToolReceipt,
            ToolRegistry,
            ToolRequirement,
        )

        return {
            "ToolDefinition": ToolDefinition,
            "ToolReceipt": ToolReceipt,
            "ToolRegistry": ToolRegistry,
            "ToolRequirement": ToolRequirement,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
