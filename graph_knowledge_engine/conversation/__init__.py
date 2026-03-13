"""Conversation-layer compatibility entrypoints.
TO-DO: Move the real file to the location if they are in other modules
Use lazy imports to avoid package-init circular imports during migration.
"""
from typing import TYPE_CHECKING
from graph_knowledge_engine.conversation.span_compat import install_span_compat_aliases

install_span_compat_aliases()


if TYPE_CHECKING:
    from graph_knowledge_engine.conversation.designer import ConversationWorkflowDesigner, AgenticAnsweringWorkflowDesigner
    from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent
    from graph_knowledge_engine.conversation.conversation_context import ConversationContextBuilder, ContextSources, PromptContext
    from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator
    from graph_knowledge_engine.conversation.conversation_state_contracts import ConversationWorkflowState
    from graph_knowledge_engine.conversation.knowledge_retriever import KnowledgeRetriever
    from graph_knowledge_engine.conversation.memory_retriever import MemoryRetriever
    from graph_knowledge_engine.conversation.retrieval_orchestrator import RetrievalOrchestrator
    from graph_knowledge_engine.conversation.service import ConversationService
    from graph_knowledge_engine.conversation.tool_runner import ToolRunner

__all__ = [
    "AgenticAnsweringAgent",
    "ConversationWorkflowDesigner", "AgenticAnsweringWorkflowDesigner",
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
]


def __getattr__(name: str):
    if name == "AgenticAnsweringAgent":
        from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent
        return AgenticAnsweringAgent
    if name in ("ConversationContextBuilder", "ContextSources", "PromptContext"):
        from graph_knowledge_engine.conversation.conversation_context import (
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
        from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator
        return ConversationOrchestrator
    if name == "WorkflowState":
        from graph_knowledge_engine.conversation.conversation_state_contracts import ConversationWorkflowState
        return ConversationWorkflowState
    if name == "KnowledgeRetriever":
        from graph_knowledge_engine.conversation.knowledge_retriever import KnowledgeRetriever
        return KnowledgeRetriever
    if name == "MemoryRetriever":
        from graph_knowledge_engine.conversation.memory_retriever import MemoryRetriever
        return MemoryRetriever
    if name == "RetrievalOrchestrator":
        from graph_knowledge_engine.conversation.retrieval_orchestrator import RetrievalOrchestrator
        return RetrievalOrchestrator
    if name == "ConversationService":
        from graph_knowledge_engine.conversation.service import ConversationService
        return ConversationService
    if name == "ToolRunner":
        from graph_knowledge_engine.conversation.tool_runner import ToolRunner
        return ToolRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
