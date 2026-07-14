"""Shared models, agent interface, and exceptions for the Supervisor Orchestrator.

Pure data + contracts. This module performs NO agent execution, NO aggregation
and NO answer generation. It defines:

* ``AgentRequest``                     — the single request object every agent receives.
* ``UniversalAgentResponse``           — a thin, faithful wrapper around one agent's output.
* ``AggregatedUniversalAgentResponse`` — the merged multi-agent result.
* ``SpecialistAgent``                  — the interface the orchestrator executes.
* ``FunctionAgentAdapter``             — a compatibility adapter that wraps an existing
                                          implementation (which may accept only a query
                                          string) as a ``SpecialistAgent``.
* the orchestrator exception hierarchy.

Design note (no rewrite of existing agents): the existing specialist agents
expose implementation-specific entry points. Rather than modify them, the
orchestrator executes objects that satisfy ``SpecialistAgent.execute(request)``.
``FunctionAgentAdapter`` bridges an existing ``callable(user_query) -> dict``
into that interface, so production can wire the real agents without changing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid a hard runtime dependency; agents are duck-typed at runtime
    from planner.planner_models import PlannerDecision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Exceptions (deterministic)
# --------------------------------------------------------------------------- #

class OrchestratorError(Exception):
    """Base error for the Supervisor Orchestrator."""


class ConfigurationError(OrchestratorError):
    """The orchestrator was constructed without a usable agent registry/engine."""


class UnknownAgentError(OrchestratorError):
    """A requested agent has no registered implementation."""


class DuplicateAgentError(OrchestratorError):
    """The planner decision lists the same agent more than once."""


class InvalidExecutionOrderError(OrchestratorError):
    """``execution_order`` is not a permutation of ``agents``."""


class PlannerMismatchError(OrchestratorError):
    """The planner decision is internally inconsistent for execution."""


class AgentExecutionError(OrchestratorError):
    """A specialist agent raised while executing (wrapped per-agent)."""


class AgentTimeoutError(OrchestratorError):
    """A specialist agent exceeded its execution timeout."""


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgentRequest:
    """The single request object constructed by the orchestrator for each agent.

    Every specialist agent receives this object — never a raw string.
    """

    request_id: str
    session_id: str
    user_query: str
    planner_decision: "PlannerDecision"
    conversation_context: dict[str, Any] | None = None
    timestamp: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "planner_decision": self.planner_decision.to_dict(),
            "conversation_context": self.conversation_context,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #

@dataclass
class UniversalAgentResponse:
    """A faithful wrapper around one specialist agent's output.

    ``payload`` is the EXACT dict the agent returned (for a successful run) or a
    minimal error envelope (for a failed run). The wrapper never edits, reorders,
    summarizes or augments the agent's evidence.
    """

    agent_name: str
    payload: dict[str, Any]
    execution_time: float = 0.0

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "UNKNOWN"))

    @property
    def error(self) -> str | None:
        value = self.payload.get("error")
        return str(value) if value else None

    def succeeded(self) -> bool:
        """True when the agent ran without error (NO_RESULTS/NO_PREDICTION count)."""
        return self.status != "ERROR" and self.error is None

    def to_dict(self) -> dict[str, Any]:
        # The agent's response is preserved exactly as returned.
        return dict(self.payload)

    @classmethod
    def error_response(
        cls, agent_name: str, message: str, execution_time: float = 0.0
    ) -> "UniversalAgentResponse":
        return cls(
            agent_name=agent_name,
            payload={"agent_name": agent_name, "status": "ERROR", "error": message},
            execution_time=execution_time,
        )


@dataclass
class AggregatedUniversalAgentResponse:
    """The merged result of executing one or more specialist agents.

    ``responses`` preserves the agent outputs exactly and in execution order.
    """

    status: str  # SUCCESS | PARTIAL_SUCCESS | FAILED | CLARIFICATION | NO_AGENTS
    request_id: str
    session_id: str
    planner_decision: dict[str, Any]
    executed_agents: list[str]
    failed_agents: list[str]
    responses: list[UniversalAgentResponse]
    execution_time: float
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "planner_decision": self.planner_decision,
            "executed_agents": list(self.executed_agents),
            "failed_agents": list(self.failed_agents),
            "clarification_question": self.clarification_question,
            "responses": [response.to_dict() for response in self.responses],
            "execution_time": round(self.execution_time, 6),
        }


# --------------------------------------------------------------------------- #
# Agent interface + compatibility adapter
# --------------------------------------------------------------------------- #

@runtime_checkable
class SpecialistAgent(Protocol):
    """The interface the orchestrator executes. Implementations return a
    ``UniversalAgentResponse`` for a given ``AgentRequest``."""

    def execute(self, request: AgentRequest) -> UniversalAgentResponse: ...


class FunctionAgentAdapter:
    """Adapts an existing implementation into a ``SpecialistAgent``.

    The existing specialist agents accept only the user query. This adapter
    exposes ``execute(request)`` and INTERNALLY calls the wrapped implementation
    with ``request.user_query``, then wraps the returned dict into a
    ``UniversalAgentResponse`` — without modifying the existing agent.

    Production wiring example (no existing code changed)::

        FunctionAgentAdapter(
            "knowledge_agent",
            lambda q: KnowledgeFormatter().format(RetrievalCoordinator().retrieve(q)),
        )
    """

    def __init__(self, agent_name: str, implementation: Callable[[str], dict[str, Any]]) -> None:
        self.agent_name = agent_name
        self._implementation = implementation

    def execute(self, request: AgentRequest) -> UniversalAgentResponse:
        start = perf_counter()
        payload = self._implementation(request.user_query)
        if not isinstance(payload, dict):
            raise AgentExecutionError(
                f"Agent '{self.agent_name}' returned {type(payload).__name__}, expected dict."
            )
        payload.setdefault("agent_name", self.agent_name)
        return UniversalAgentResponse(
            agent_name=self.agent_name,
            payload=payload,
            execution_time=perf_counter() - start,
        )
