"""Conversation Memory Manager package for the AquaMind AI Supervisor Agent.

The first component executed on every user request and the last one updated.
Stores and retrieves conversation state only -- independent of every LLM and
every specialist agent.
"""

from .conversation_memory import DEFAULT_SHORT_TERM_LIMIT, ConversationMemory
from .memory_manager import LongTermMemoryStore, MemoryManager, SessionNotFoundError
from .models import (
    AgentName,
    AgentResponseRef,
    ConversationContext,
    ConversationMetadata,
    IntentType,
    Message,
    MessageRole,
    ReferencedEntities,
)
from .session import Session

__all__ = [
    "ConversationMemory",
    "DEFAULT_SHORT_TERM_LIMIT",
    "MemoryManager",
    "LongTermMemoryStore",
    "SessionNotFoundError",
    "Session",
    "Message",
    "MessageRole",
    "AgentName",
    "IntentType",
    "ConversationContext",
    "ReferencedEntities",
    "AgentResponseRef",
    "ConversationMetadata",
]
