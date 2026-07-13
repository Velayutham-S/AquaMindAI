"""Single-conversation session state for the AquaMind AI Conversation Memory.

A :class:`Session` owns the complete state of exactly ONE conversation: its
message history (short-term memory), current context, referenced entities, turn
counter and timestamps. Each session carries its OWN re-entrant lock, so
concurrent access to different sessions never contends and one conversation can
never affect another (session isolation).

This module stores and retrieves state only. It performs NO reasoning, NO intent
classification, NO LLM calls, NO summarization and NO entity inference.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import (
    AgentResponseRef,
    ConversationContext,
    ConversationMetadata,
    Message,
    MessageRole,
    ReferencedEntities,
)


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    """State for one conversation. Thread-safe for all of its own operations."""

    session_id: str
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    context: ConversationContext = field(default_factory=ConversationContext)
    entities: ReferencedEntities = field(default_factory=ReferencedEntities)
    metadata: ConversationMetadata = field(default_factory=ConversationMetadata)
    last_response: AgentResponseRef | None = None
    _turn: int = 0
    _history: list[Message] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Messages (short-term memory)
    # ------------------------------------------------------------------ #

    def add_message(self, role: MessageRole, content: str) -> Message:
        """Append a message tagged with the current turn; returns it."""
        with self._lock:
            message = Message(role=role, content=content, timestamp=_utc_now(), turn=self._turn)
            self._history.append(message)
            self._touch()
            return message

    def get_recent_messages(self, limit: int) -> list[Message]:
        """Return a copy of the most recent messages (all if ``limit`` <= 0)."""
        with self._lock:
            if limit <= 0:
                return list(self._history)
            return list(self._history[-limit:])

    def get_last_message(self, role: MessageRole) -> Message | None:
        """Return the most recent message with the given role, or ``None``."""
        with self._lock:
            for message in reversed(self._history):
                if message.role == role:
                    return message
            return None

    def message_count(self) -> int:
        with self._lock:
            return len(self._history)

    # ------------------------------------------------------------------ #
    # Context / entities
    # ------------------------------------------------------------------ #

    def update_context(self, updates: dict[str, Any]) -> ConversationContext:
        with self._lock:
            self.context.apply(updates)
            self._touch()
            return self.context

    def get_context(self) -> ConversationContext:
        with self._lock:
            return self.context

    def update_entities(self, updates: dict[str, Any]) -> ReferencedEntities:
        with self._lock:
            self.entities.apply(updates)
            self._touch()
            return self.entities

    def get_entities(self) -> ReferencedEntities:
        with self._lock:
            return self.entities

    # ------------------------------------------------------------------ #
    # Last response reference (lightweight pointer to prior agent output)
    # ------------------------------------------------------------------ #

    def set_last_response(self, response_ref: AgentResponseRef) -> AgentResponseRef:
        with self._lock:
            self.last_response = response_ref
            self._touch()
            return self.last_response

    def get_last_response(self) -> AgentResponseRef | None:
        with self._lock:
            return self.last_response

    # ------------------------------------------------------------------ #
    # Conversation metadata (session-level preferences)
    # ------------------------------------------------------------------ #

    def update_metadata(self, updates: dict[str, Any]) -> ConversationMetadata:
        with self._lock:
            self.metadata.apply(updates)
            self._touch()
            return self.metadata

    def get_metadata(self) -> ConversationMetadata:
        with self._lock:
            return self.metadata

    # ------------------------------------------------------------------ #
    # Turns
    # ------------------------------------------------------------------ #

    def increment_turn(self) -> int:
        with self._lock:
            self._turn += 1
            self._touch()
            return self._turn

    def get_turn(self) -> int:
        with self._lock:
            return self._turn

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Reset conversation state (history, turn, context, entities and the
        last-response pointer) but keep the session alive under the same id.

        Session-level ``metadata`` (language / timezone / preferred units) is a
        user preference and is intentionally PRESERVED across a clear.
        """
        with self._lock:
            self._history.clear()
            self._turn = 0
            self.context = ConversationContext()
            self.entities = ReferencedEntities()
            self.last_response = None
            self._touch()

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the whole session (excludes the lock)."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "turn": self._turn,
                "message_count": len(self._history),
                "history": [message.to_dict() for message in self._history],
                "context": self.context.to_dict(),
                "entities": self.entities.to_dict(),
                "metadata": self.metadata.to_dict(),
                "last_response": self.last_response.to_dict() if self.last_response else None,
            }

    def _touch(self) -> None:
        """Record the last-modified time. Must be called while holding the lock."""
        self.updated_at = _utc_now()
