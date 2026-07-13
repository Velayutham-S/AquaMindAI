"""Public Conversation Memory API for AquaMind AI.

:class:`ConversationMemory` is the single entry point the future Supervisor
Planner will use. It is the FIRST component executed on every user request and
the LAST component updated. It stores and retrieves conversation state only --
it performs NO reasoning, NO intent classification, NO routing, NO LLM calls, NO
answer generation, NO summarization and NO entity inference.

Design: it is a thin facade that delegates session lifecycle to an injected
:class:`MemoryManager` and per-session operations to the :class:`Session`
objects that manager owns. This keeps responsibilities separate:

* ``Session``          -- state of one conversation (+ its own lock)
* ``MemoryManager``    -- registry/coordinator of all active sessions
* ``ConversationMemory`` -- the unified public API over both
"""

from __future__ import annotations

from typing import Any

from .memory_manager import MemoryManager
from .models import (
    AgentResponseRef,
    ConversationContext,
    ConversationMetadata,
    Message,
    MessageRole,
    ReferencedEntities,
)
from .session import Session

#: Default number of most-recent messages returned as "short-term" memory.
DEFAULT_SHORT_TERM_LIMIT = 20


class ConversationMemory:
    """Stores and retrieves conversation state for many isolated sessions."""

    def __init__(
        self,
        manager: MemoryManager | None = None,
        short_term_limit: int = DEFAULT_SHORT_TERM_LIMIT,
    ) -> None:
        self._manager = manager or MemoryManager()
        self._short_term_limit = short_term_limit

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def create_session(self, session_id: str | None = None) -> Session:
        return self._manager.create_session(session_id)

    def session_exists(self, session_id: str) -> bool:
        return self._manager.session_exists(session_id)

    def get_session(self, session_id: str) -> Session:
        return self._manager.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._manager.delete_session(session_id)

    def clear_session(self, session_id: str) -> None:
        self._manager.get_session(session_id).clear()

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #

    def add_user_message(self, session_id: str, content: str) -> Message:
        return self._manager.get_session(session_id).add_message(MessageRole.USER, content)

    def add_assistant_message(self, session_id: str, content: str) -> Message:
        return self._manager.get_session(session_id).add_message(MessageRole.ASSISTANT, content)

    def get_recent_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Return the most recent messages (short-term memory).

        ``limit=None`` uses the configured short-term window; a positive
        ``limit`` returns the last ``limit`` messages.
        """
        window = self._short_term_limit if limit is None else limit
        return self._manager.get_session(session_id).get_recent_messages(window)

    def get_last_user_message(self, session_id: str) -> Message | None:
        return self._manager.get_session(session_id).get_last_message(MessageRole.USER)

    def get_last_assistant_message(self, session_id: str) -> Message | None:
        return self._manager.get_session(session_id).get_last_message(MessageRole.ASSISTANT)

    # ------------------------------------------------------------------ #
    # Context (follow-up state -- stored, never inferred)
    # ------------------------------------------------------------------ #

    def update_context(self, session_id: str, **updates: Any) -> ConversationContext:
        """Merge the provided context fields (e.g. ``current_district='Salem'``)."""
        return self._manager.get_session(session_id).update_context(updates)

    def get_context(self, session_id: str) -> ConversationContext:
        return self._manager.get_session(session_id).get_context()

    # ------------------------------------------------------------------ #
    # Referenced entities (stored, never inferred)
    # ------------------------------------------------------------------ #

    def update_entities(self, session_id: str, **updates: Any) -> ReferencedEntities:
        return self._manager.get_session(session_id).update_entities(updates)

    def get_entities(self, session_id: str) -> ReferencedEntities:
        return self._manager.get_session(session_id).get_entities()

    # ------------------------------------------------------------------ #
    # Last response reference (stored, never inferred)
    # ------------------------------------------------------------------ #

    def set_last_response(
        self,
        session_id: str,
        response_id: str,
        timestamp: str,
        agent_names: "list[str] | tuple[str, ...]" = (),
        status: str | None = None,
    ) -> AgentResponseRef:
        """Store a lightweight pointer to the most recent agent response."""
        response_ref = AgentResponseRef(
            response_id=response_id,
            timestamp=timestamp,
            agent_names=tuple(agent_names),
            status=status,
        )
        return self._manager.get_session(session_id).set_last_response(response_ref)

    def get_last_response(self, session_id: str) -> AgentResponseRef | None:
        return self._manager.get_session(session_id).get_last_response()

    # ------------------------------------------------------------------ #
    # Conversation metadata (session-level preferences, stored not inferred)
    # ------------------------------------------------------------------ #

    def update_metadata(self, session_id: str, **updates: Any) -> ConversationMetadata:
        return self._manager.get_session(session_id).update_metadata(updates)

    def get_metadata(self, session_id: str) -> ConversationMetadata:
        return self._manager.get_session(session_id).get_metadata()

    # ------------------------------------------------------------------ #
    # Turns
    # ------------------------------------------------------------------ #

    def increment_turn(self, session_id: str) -> int:
        return self._manager.get_session(session_id).increment_turn()

    def get_turn(self, session_id: str) -> int:
        return self._manager.get_session(session_id).get_turn()

    # ------------------------------------------------------------------ #
    # Coordination
    # ------------------------------------------------------------------ #

    @property
    def manager(self) -> MemoryManager:
        """The underlying session coordinator (registry of active sessions)."""
        return self._manager
