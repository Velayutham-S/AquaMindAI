"""Supervisor Orchestrator package for AquaMind AI.

Executes the specialist agents chosen by the Supervisor Planner and merges their
responses. It orchestrates only — it never generates answers, calls the
Recommendation/Response Generator LLMs, or modifies evidence.
"""

from .execution_context import ExecutionContext
from .execution_engine import ExecutionEngine
from .orchestrator import SupervisorOrchestrator
from .request_models import (
    AgentExecutionError,
    AgentRequest,
    AgentTimeoutError,
    AggregatedUniversalAgentResponse,
    ConfigurationError,
    DuplicateAgentError,
    FunctionAgentAdapter,
    InvalidExecutionOrderError,
    OrchestratorError,
    PlannerMismatchError,
    SpecialistAgent,
    UniversalAgentResponse,
    UnknownAgentError,
)
from .response_aggregator import ResponseAggregator

__all__ = [
    "SupervisorOrchestrator",
    "ExecutionEngine",
    "ExecutionContext",
    "ResponseAggregator",
    "AgentRequest",
    "UniversalAgentResponse",
    "AggregatedUniversalAgentResponse",
    "SpecialistAgent",
    "FunctionAgentAdapter",
    "OrchestratorError",
    "ConfigurationError",
    "UnknownAgentError",
    "DuplicateAgentError",
    "InvalidExecutionOrderError",
    "PlannerMismatchError",
    "AgentExecutionError",
    "AgentTimeoutError",
]
