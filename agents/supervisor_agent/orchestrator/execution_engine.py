"""Execution Engine for the Supervisor Orchestrator.

Single responsibility: execute the selected specialist agents STRICTLY in the
planner's ``execution_order`` and collect one ``UniversalAgentResponse`` per
agent. It never reorders agents, never generates answers, and never aggregates.

Resilience: if one agent is unknown, times out, or raises, the failure is
recorded (an error ``UniversalAgentResponse`` is appended and the agent is
marked failed) and execution continues with the remaining agents.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from time import perf_counter

from .execution_context import ExecutionContext
from .request_models import (
    AgentRequest,
    AgentTimeoutError,
    SpecialistAgent,
    UnknownAgentError,
    UniversalAgentResponse,
)

logger = logging.getLogger("aquamind.supervisor.execution_engine")


class ExecutionEngine:
    """Runs specialist agents in order over an injected agent registry."""

    def __init__(
        self,
        agents: dict[str, SpecialistAgent],
        agent_timeout: float | None = None,
    ) -> None:
        # Dependency injection: the engine knows nothing about concrete agents.
        self._agents = dict(agents)
        self._agent_timeout = agent_timeout

    @property
    def registered_agents(self) -> tuple[str, ...]:
        return tuple(self._agents.keys())

    def execute(
        self, request: AgentRequest, context: ExecutionContext
    ) -> list[UniversalAgentResponse]:
        """Execute each agent in ``execution_order``; return responses in that order."""
        responses: list[UniversalAgentResponse] = []
        for agent in request.planner_decision.execution_order:
            agent_name = agent.value if hasattr(agent, "value") else str(agent)
            response = self._execute_one(agent_name, request)
            responses.append(response)
            if response.succeeded():
                context.record_executed(agent_name)
            else:
                context.record_failed(agent_name)
        return responses

    def _execute_one(self, agent_name: str, request: AgentRequest) -> UniversalAgentResponse:
        """Execute a single agent, converting any failure into an error response."""
        start = perf_counter()
        try:
            agent = self._agents.get(agent_name)
            if agent is None:
                raise UnknownAgentError(f"No agent registered for '{agent_name}'.")
            return self._run_with_optional_timeout(agent_name, agent, request)
        except UnknownAgentError as error:
            logger.warning("Unknown agent '%s': %s", agent_name, error)
            return UniversalAgentResponse.error_response(
                agent_name, str(error), execution_time=perf_counter() - start
            )
        except AgentTimeoutError as error:
            logger.warning("Agent '%s' timed out: %s", agent_name, error)
            return UniversalAgentResponse.error_response(
                agent_name, str(error), execution_time=perf_counter() - start
            )
        except Exception as error:  # noqa: BLE001 - any agent failure is recorded, not fatal
            logger.warning("Agent '%s' raised %s: %s", agent_name, type(error).__name__, error)
            return UniversalAgentResponse.error_response(
                agent_name,
                f"{type(error).__name__}: {error}",
                execution_time=perf_counter() - start,
            )

    def _run_with_optional_timeout(
        self, agent_name: str, agent: SpecialistAgent, request: AgentRequest
    ) -> UniversalAgentResponse:
        if not self._agent_timeout or self._agent_timeout <= 0:
            return agent.execute(request)
        # Enforce a per-agent wall-clock timeout without blocking other agents.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(agent.execute, request)
            try:
                return future.result(timeout=self._agent_timeout)
            except FuturesTimeoutError as error:
                raise AgentTimeoutError(
                    f"Agent '{agent_name}' exceeded timeout of {self._agent_timeout}s."
                ) from error
