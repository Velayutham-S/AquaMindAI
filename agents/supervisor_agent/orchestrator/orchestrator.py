"""Supervisor Orchestrator for AquaMind AI.

Single responsibility: given a validated ``PlannerDecision`` and the user query,
construct one ``AgentRequest``, execute the selected specialist agents in the
planner's execution order, collect their ``UniversalAgentResponse`` objects, and
return one ``AggregatedUniversalAgentResponse``.

It NEVER generates natural-language answers, NEVER calls the Recommendation or
Response Generator LLMs, NEVER summarizes, and NEVER modifies evidence. It only
orchestrates execution and merges the results.

Dependency injection: the orchestrator is given an agent registry (or a
pre-built execution engine) plus an aggregator, so it is fully decoupled from the
concrete specialist agents and is deterministic to test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .execution_context import ExecutionContext
from .execution_engine import ExecutionEngine
from .request_models import (
    AgentRequest,
    AggregatedUniversalAgentResponse,
    ConfigurationError,
    DuplicateAgentError,
    InvalidExecutionOrderError,
    PlannerMismatchError,
    SpecialistAgent,
)
from .response_aggregator import ResponseAggregator

if TYPE_CHECKING:
    from planner.planner_models import PlannerDecision


class SupervisorOrchestrator:
    """Executes the planner's decision and aggregates specialist responses."""

    def __init__(
        self,
        agents: dict[str, SpecialistAgent] | None = None,
        engine: ExecutionEngine | None = None,
        aggregator: ResponseAggregator | None = None,
        agent_timeout: float | None = None,
    ) -> None:
        if engine is None:
            if agents is None:
                raise ConfigurationError("Provide either an agent registry or an execution engine.")
            engine = ExecutionEngine(agents, agent_timeout=agent_timeout)
        self._engine = engine
        self._aggregator = aggregator or ResponseAggregator()

    def orchestrate(
        self,
        planner_decision: "PlannerDecision",
        user_query: str,
        session_id: str,
        conversation_context: dict[str, Any] | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AggregatedUniversalAgentResponse:
        """Run the decision and return the aggregated response."""
        request = AgentRequest(
            request_id=request_id or uuid4().hex,
            session_id=session_id,
            user_query=user_query,
            planner_decision=planner_decision,
            conversation_context=conversation_context,
            metadata=metadata or {},
        )

        # Clarification short-circuit: nothing executes.
        if planner_decision.requires_clarification:
            return self._aggregator.clarification(
                request, planner_decision.clarification_question
            )

        self._validate_decision(planner_decision)

        context = ExecutionContext(
            request_id=request.request_id,
            session_id=request.session_id,
            planner_decision=planner_decision.to_dict(),
        )
        context.start()
        responses = self._engine.execute(request, context)
        context.finish()

        return self._aggregator.aggregate(request, responses, context)

    @staticmethod
    def _validate_decision(planner_decision: "PlannerDecision") -> None:
        """Deterministic pre-flight validation of an executable decision."""
        agents = [a.value if hasattr(a, "value") else str(a) for a in planner_decision.agents]
        order = [
            a.value if hasattr(a, "value") else str(a)
            for a in planner_decision.execution_order
        ]

        if not agents:
            raise PlannerMismatchError(
                "Planner decision requires execution but selected no agents."
            )
        if len(set(agents)) != len(agents):
            raise DuplicateAgentError(f"Duplicate agent(s) in 'agents': {agents}.")
        if len(set(order)) != len(order):
            raise DuplicateAgentError(f"Duplicate agent(s) in 'execution_order': {order}.")
        if sorted(agents) != sorted(order):
            raise InvalidExecutionOrderError(
                f"execution_order {order} is not a permutation of agents {agents}."
            )
