"""Conversation service façade following service-pattern composition.

ConversationService
├─ uses GraphKnowledgeEngine
├─ uses WorkflowRuntime
├─ uses LLM
├─ uses knowledge engine
└─ implements conversation behavior
"""

from __future__ import annotations

from typing import Any, Optional

from graph_knowledge_engine.engine_core import GraphKnowledgeEngine
from graph_knowledge_engine.runtime import WorkflowRuntime

from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator


class ConversationService:
    """High-level conversation behavior service.

    This provides a stable service-pattern entrypoint while preserving current
    orchestrator behavior internally.
    """

    def __init__(
        self,
        *,
        conversation_engine: GraphKnowledgeEngine,
        knowledge_engine: GraphKnowledgeEngine,
        workflow_engine: Optional[GraphKnowledgeEngine] = None,
        llm: Any | None = None,
        runtime_cls: type[WorkflowRuntime] = WorkflowRuntime,
    ) -> None:
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine
        self.workflow_engine = workflow_engine
        self.llm = llm or conversation_engine.llm
        self.runtime_cls = runtime_cls

        self.orchestrator = ConversationOrchestrator(
            conversation_engine=conversation_engine,
            ref_knowledge_engine=knowledge_engine,
            workflow_engine=workflow_engine,
            llm=self.llm,
        )

    def create_conversation(self, *args, **kwargs):
        """Create conversation via engine primitive.

        Keep this independent from orchestrator delegation to avoid recursion when
        engine-level compatibility shims call into ConversationService.
        """
        create_primitive = getattr(self.conversation_engine, "_create_conversation_primitive", None)
        if callable(create_primitive):
            return create_primitive(*args, **kwargs)
        return self.orchestrator.create_conversation(*args, **kwargs)

    def add_turn(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn(*args, **kwargs)

    # Compatibility names used by legacy engine shims
    def add_conversation_turn(self, *args, **kwargs):
        return self.add_turn(*args, **kwargs)

    def add_turn_workflow_v2(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn_workflow_v2(*args, **kwargs)

    def answer_only(self, *args, **kwargs):
        return self.orchestrator.answer_only(*args, **kwargs)

    def get_ai_conversation_response(self, *args, **kwargs):
        return self.answer_only(*args, **kwargs)
