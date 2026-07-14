"""Per-execution context for the Supervisor Orchestrator.

``ExecutionContext`` holds the mutable bookkeeping for a single orchestration:
timing plus which agents executed or failed. It exists only during execution and
is consumed by the Response Aggregator. It performs no agent execution and no
aggregation itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class ExecutionContext:
    """Mutable state tracked across one orchestration run."""

    request_id: str
    session_id: str
    planner_decision: dict[str, Any]
    execution_start: float = 0.0
    execution_end: float = 0.0
    executed_agents: list[str] = field(default_factory=list)
    failed_agents: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.execution_start = perf_counter()

    def finish(self) -> None:
        self.execution_end = perf_counter()

    def record_executed(self, agent_name: str) -> None:
        self.executed_agents.append(agent_name)

    def record_failed(self, agent_name: str) -> None:
        self.failed_agents.append(agent_name)

    @property
    def execution_time(self) -> float:
        return max(self.execution_end - self.execution_start, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "planner_decision": self.planner_decision,
            "executed_agents": list(self.executed_agents),
            "failed_agents": list(self.failed_agents),
            "execution_time": round(self.execution_time, 6),
        }
