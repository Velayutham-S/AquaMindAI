"""Data models for the AquaMind AI Conversation Memory Manager.

Pure, framework-free data holders. They deliberately know NOTHING about LLMs,
specialist agents, SQL, RAG, embeddings, prediction models or response
generation -- they only describe conversation state so it can be stored and
retrieved. All models are designed to be easily extended (Open-Closed): new
context fields can be added, and unknown fields land in an ``extra`` bag rather
than being rejected.

This module performs NO reasoning, NO intent classification, NO summarization
and NO entity inference. It is data only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Author of a conversation message. ``str`` mixin -> clean serialization."""

    USER = "user"
    ASSISTANT = "assistant"


class AgentName(str, Enum):
    """Recommended vocabulary for the currently/previously active agent.

    Reference values only -- the Conversation Memory Manager never assigns these;
    the future Supervisor writes them. Stored as plain strings, so new agents can
    be recorded without changing this enum.
    """

    DATA_AGENT = "data_agent"
    KNOWLEDGE_AGENT = "knowledge_agent"
    PREDICTION_AGENT = "prediction_agent"
    GENERAL_LLM = "general_llm"


class IntentType(str, Enum):
    """Recommended vocabulary for the current conversation intent.

    Reference values only -- written by the future Supervisor, never inferred
    here. Stored as plain strings for forward compatibility.
    """

    DATA_QUERY = "data_query"
    KNOWLEDGE_QUERY = "knowledge_query"
    PREDICTION_QUERY = "prediction_query"
    GENERAL_CHAT = "general_chat"
    SYSTEM_INFORMATION = "system_information"


@dataclass(frozen=True)
class Message:
    """A single conversation message (immutable, safe to share across readers)."""

    role: MessageRole
    content: str
    timestamp: str  # ISO-8601 UTC
    turn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "turn": self.turn,
        }


@dataclass
class ConversationContext:
    """The current, follow-up-relevant conversation context.

    Every field is a plain stored value. The Conversation Memory Manager NEVER
    infers these -- the future Supervisor Planner reads and updates them. Fields
    default to ``None`` (unknown). Unknown keys passed to :meth:`apply` are kept
    in ``extra`` so new context can be added without changing this class.
    """

    # --- topics ---
    current_topic: str | None = None
    current_conversation_topic: str | None = None
    current_data_topic: str | None = None
    current_knowledge_topic: str | None = None
    current_groundwater_topic: str | None = None
    current_prediction_target: str | None = None

    # --- routing / intent (written by the Supervisor; NEVER inferred here) ---
    #: See AgentName for the recommended vocabulary (e.g. "prediction_agent").
    current_active_agent: str | None = None
    #: See IntentType for the recommended vocabulary (e.g. "prediction_query").
    current_intent: str | None = None

    # --- geography ---
    current_district: str | None = None
    current_taluk: str | None = None
    current_firka: str | None = None
    current_village: str | None = None

    # --- time ---
    current_year: int | str | None = None
    current_month: int | str | None = None

    # --- reserved placeholder: summarization is NOT implemented in this milestone ---
    conversation_summary: str | None = None

    # --- forward-compatible bag for future context fields ---
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(self, updates: dict[str, Any]) -> None:
        """Set the provided keys. Known fields are set directly; the rest go to
        ``extra``. Only keys present in ``updates`` are touched (merge semantics).
        """
        known = {f.name for f in fields(self)} - {"extra"}
        for key, value in updates.items():
            if key in known:
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferencedEntities:
    """Entities referenced during the conversation (stored, never inferred).

    Backed by a flexible mapping so any entity kind (districts, years, targets,
    ...) can be tracked without schema changes.
    """

    entities: dict[str, Any] = field(default_factory=dict)

    def apply(self, updates: dict[str, Any]) -> None:
        """Merge the provided entity values (last write wins per key)."""
        self.entities.update(updates)

    def get(self, name: str, default: Any = None) -> Any:
        return self.entities.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.entities)


@dataclass(frozen=True)
class AgentResponseRef:
    """A lightweight reference to a prior UniversalAgentResponse.

    Deliberately stores only the small, identifying fields (not the full
    response payload), so the Supervisor can know where the last answer came
    from without re-running anything. Immutable.
    """

    response_id: str
    timestamp: str  # ISO-8601 UTC
    agent_names: tuple[str, ...] = ()
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "timestamp": self.timestamp,
            "agent_names": list(self.agent_names),
            "status": self.status,
        }


@dataclass
class ConversationMetadata:
    """Session-level metadata / user preferences (future-proof).

    Written by the Supervisor / application layer, never inferred here. Unknown
    keys are kept in ``extra`` so new metadata can be added freely.
    """

    language: str | None = None
    timezone: str | None = None
    preferred_units: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(self, updates: dict[str, Any]) -> None:
        known = {f.name for f in fields(self)} - {"extra"}
        for key, value in updates.items():
            if key in known:
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
