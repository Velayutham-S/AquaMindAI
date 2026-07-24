"""Strongly-typed models and exceptions for the AquaMind AI Supervisor Planner.

Pure data + error definitions. This module performs NO reasoning, NO LLM calls,
and NO agent execution. It only describes:

* the controlled vocabularies the Planner emits (intent, confidence, agent),
* the validated routing decision (``PlannerDecision``),
* the Planner's exception hierarchy.

Scope note: in this milestone the Planner may route only to the three specialist
agents (data / knowledge / prediction). The General Conversation LLM is NOT part
of this milestone and is therefore NOT a valid ``AgentName``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    """Classified intent of the user query (``str`` mixin -> clean JSON)."""

    DATA_QUERY = "data_query"
    KNOWLEDGE_QUERY = "knowledge_query"
    PREDICTION_QUERY = "prediction_query"
    SYSTEM_INFORMATION = "system_information"
    MIXED_QUERY = "mixed_query"
    # Out-of-domain intents: the query is NOT a groundwater request AquaMind AI
    # handles (greetings / small talk -> general_chat; any other non-groundwater
    # or disallowed request -> out_of_scope). These select NO specialist agent;
    # the pipeline returns the predefined out-of-domain message (no LLM is called).
    GENERAL_CHAT = "general_chat"
    OUT_OF_SCOPE = "out_of_scope"


class ConfidenceLevel(str, Enum):
    """Routing confidence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AgentName(str, Enum):
    """The only specialist agents the Planner may select in this milestone."""

    DATA_AGENT = "data_agent"
    KNOWLEDGE_AGENT = "knowledge_agent"
    PREDICTION_AGENT = "prediction_agent"


#: Intents that mark a query as OUT OF DOMAIN (outside AquaMind AI's groundwater
#: focus). An out-of-domain decision is terminal: it selects no specialist agent,
#: triggers no LLM call, and the pipeline returns the predefined application
#: message instead of running the groundwater workflow.
OUT_OF_DOMAIN_INTENTS: frozenset[IntentType] = frozenset(
    {IntentType.GENERAL_CHAT, IntentType.OUT_OF_SCOPE}
)


@dataclass(frozen=True)
class PlannerDecision:
    """A validated routing decision produced by the Supervisor Planner.

    Immutable. Construct only via the validator, which guarantees every field is
    consistent (enums, allowed agents, execution order, clarification state).
    """

    intent: IntentType
    confidence: ConfidenceLevel
    requires_clarification: bool
    clarification_question: str | None
    agents: tuple[AgentName, ...]
    execution_order: tuple[AgentName, ...]
    reason: str
    # Structured prediction slots. Populated ONLY for prediction / mixed
    # prediction queries; null for every other intent. The Planner extracts
    # these from the query (or resolves them from Conversation Memory); the
    # Prediction Agent consumes them as structured inputs and performs no NLU.
    district: str | None = None
    firka: str | None = None
    target_year: int | None = None

    @property
    def is_out_of_domain(self) -> bool:
        """True when the query falls outside AquaMind AI's groundwater domain.

        Out-of-domain decisions run no specialist agent; the pipeline returns the
        predefined out-of-domain message without calling any LLM.
        """
        return self.intent in OUT_OF_DOMAIN_INTENTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence.value,
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
            "agents": [agent.value for agent in self.agents],
            "execution_order": [agent.value for agent in self.execution_order],
            "district": self.district,
            "firka": self.firka,
            "target_year": self.target_year,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #

class PlannerError(Exception):
    """Base error for the Supervisor Planner."""


class ConfigurationError(PlannerError):
    """Missing/invalid configuration (e.g. API key)."""


class InputFileError(PlannerError):
    """A required planner input file is missing or unreadable."""


class EmptyQueryError(PlannerError):
    """The user query was empty or blank."""


class PlannerApiError(PlannerError):
    """The Planner LLM API call failed (after retries) or returned no text."""


class PlannerResponseError(PlannerError):
    """The Planner LLM response was not parseable as a single JSON object.

    The offending raw text is attached as ``.raw`` for diagnostics.
    """

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class PlannerValidationError(PlannerError):
    """The routing JSON failed validation (missing/invalid/inconsistent fields).

    The offending payload is attached as ``.payload`` for diagnostics.
    """

    def __init__(self, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload
