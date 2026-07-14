"""Response Aggregator for the Supervisor Orchestrator.

Single responsibility: merge the individual ``UniversalAgentResponse`` objects
produced during one orchestration into a single
``AggregatedUniversalAgentResponse``.

It ONLY merges. It does NOT generate new information, summarize, call any LLM,
or modify / remove / reorder / re-rank evidence. Each agent's response is carried
through exactly as returned, in execution order.
"""

from __future__ import annotations

from .execution_context import ExecutionContext
from .request_models import (
    AgentRequest,
    AggregatedUniversalAgentResponse,
    UniversalAgentResponse,
)

STATUS_SUCCESS = "SUCCESS"
STATUS_PARTIAL = "PARTIAL_SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_NO_AGENTS = "NO_AGENTS"
STATUS_CLARIFICATION = "CLARIFICATION"


class ResponseAggregator:
    """Merges specialist responses into one aggregated response (no reasoning)."""

    def aggregate(
        self,
        request: AgentRequest,
        responses: list[UniversalAgentResponse],
        context: ExecutionContext,
    ) -> AggregatedUniversalAgentResponse:
        """Combine responses using only the execution bookkeeping in ``context``."""
        status = self._derive_status(context, responses)
        return AggregatedUniversalAgentResponse(
            status=status,
            request_id=request.request_id,
            session_id=request.session_id,
            planner_decision=request.planner_decision.to_dict(),
            executed_agents=list(context.executed_agents),
            failed_agents=list(context.failed_agents),
            responses=list(responses),  # preserved exactly, in execution order
            execution_time=context.execution_time,
        )

    def clarification(
        self, request: AgentRequest, clarification_question: str | None
    ) -> AggregatedUniversalAgentResponse:
        """Build the aggregated response for a clarification (no agents executed)."""
        return AggregatedUniversalAgentResponse(
            status=STATUS_CLARIFICATION,
            request_id=request.request_id,
            session_id=request.session_id,
            planner_decision=request.planner_decision.to_dict(),
            executed_agents=[],
            failed_agents=[],
            responses=[],
            execution_time=0.0,
            clarification_question=clarification_question,
        )

    @staticmethod
    def _derive_status(
        context: ExecutionContext, responses: list[UniversalAgentResponse]
    ) -> str:
        if not responses:
            return STATUS_NO_AGENTS
        if not context.failed_agents:
            return STATUS_SUCCESS
        if context.executed_agents:
            return STATUS_PARTIAL
        return STATUS_FAILED
