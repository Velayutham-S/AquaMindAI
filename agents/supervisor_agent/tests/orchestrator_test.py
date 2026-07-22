"""Temporary integration test for the AquaMind AI Supervisor Orchestrator.

NOT production code. It verifies the orchestration pipeline deterministically:

    PlannerDecision
      -> SupervisorOrchestrator
      -> ExecutionEngine (agents run in execution_order)
      -> UniversalAgentResponse(s)
      -> ResponseAggregator
      -> AggregatedUniversalAgentResponse

To keep the test deterministic and fast (per the orchestrator's design goal),
the specialist agents are replaced with lightweight, fully-controllable doubles
that return realistic UniversalAgentResponse payloads. This exercises the
orchestration logic (execution order, aggregation, clarification, failure
continuation, unknown agent) WITHOUT calling any real LLM / FAISS / model.

Results are written incrementally to ``orchestrator_result.txt`` (flushed per
case) so nothing is lost if the run is interrupted.

Run:
    python agents/supervisor_agent/tests/orchestrator_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

TEST_DIR = Path(__file__).resolve().parent
SUPERVISOR_DIR = TEST_DIR.parent
PROJECT_ROOT = TEST_DIR.parents[2]  # tests -> supervisor_agent -> agents -> root
if str(SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from orchestrator import (  # noqa: E402
    AgentRequest,
    DuplicateAgentError,
    InvalidExecutionOrderError,
    PlannerMismatchError,
    SupervisorOrchestrator,
    UniversalAgentResponse,
)
from planner.planner_models import (  # noqa: E402
    AgentName,
    ConfidenceLevel,
    IntentType,
    PlannerDecision,
)

BENCHMARK_PATH = SUPERVISOR_DIR / "orchestrator_benchmark.json"
OUTPUT_PATH = PROJECT_ROOT / "orchestrator_result.txt"
SECTION = "=" * 50
ALL_AGENTS = ("data_agent", "knowledge_agent", "prediction_agent")


# --------------------------------------------------------------------------- #
# Deterministic agent doubles (test only)
# --------------------------------------------------------------------------- #

def _default_payload(agent_name: str, request: AgentRequest) -> dict:
    """A realistic UniversalAgentResponse payload per agent type."""
    if agent_name == "data_agent":
        return {
            "agent_name": "data_agent",
            "status": "SUCCESS",
            "query_type": "data",
            "row_count": 1,
            "evidence": [{"query": request.user_query, "value": 6.55, "unit": "m"}],
        }
    if agent_name == "knowledge_agent":
        return {
            "agent_name": "knowledge_agent",
            "status": "SUCCESS",
            "query_type": "knowledge",
            "retrieval_method": "semantic_search",
            "total_evidence": 1,
            "evidence": [{"document": "sample.pdf", "page": 1, "similarity_score": 0.78,
                          "content": "sample evidence"}],
        }
    return {
        "agent_name": "prediction_agent",
        "status": "SUCCESS",
        "query_type": "prediction",
        "prediction_method": "machine_learning",
        "model_name": "XGBoost",
        "prediction": {"target": "groundwater_level_m", "predicted_value": -5.92,
                       "unit": "metres below ground level"},
    }


class FakeAgent:
    """A deterministic SpecialistAgent double."""

    def __init__(self, agent_name: str, *, fail: bool = False) -> None:
        self.agent_name = agent_name
        self._fail = fail

    def execute(self, request: AgentRequest) -> UniversalAgentResponse:
        if self._fail:
            raise RuntimeError(f"{self.agent_name} simulated failure")
        return UniversalAgentResponse(
            agent_name=self.agent_name,
            payload=_default_payload(self.agent_name, request),
        )


def _build_registry(fail_agents: set[str], missing_agents: set[str]) -> dict:
    registry = {}
    for name in ALL_AGENTS:
        if name in missing_agents:
            continue
        registry[name] = FakeAgent(name, fail=name in fail_agents)
    return registry


def _build_decision(raw: dict) -> PlannerDecision:
    return PlannerDecision(
        intent=IntentType(raw["intent"]),
        confidence=ConfidenceLevel(raw["confidence"]),
        requires_clarification=raw["requires_clarification"],
        clarification_question=raw.get("clarification_question"),
        agents=tuple(AgentName(a) for a in raw["agents"]),
        execution_order=tuple(AgentName(a) for a in raw["execution_order"]),
        reason=raw["reason"],
    )


# --------------------------------------------------------------------------- #
# Expectation checking
# --------------------------------------------------------------------------- #

def _evaluate(entry: dict, aggregated) -> tuple[bool, str]:
    """Return (passed, note) comparing the aggregated result to expectations."""
    decision = entry["decision"]
    order = list(decision["execution_order"])
    fail = set(entry.get("fail_agents", []))
    missing = set(entry.get("missing_agents", []))

    if decision["requires_clarification"]:
        ok = (aggregated.status == "CLARIFICATION"
              and aggregated.executed_agents == []
              and aggregated.responses == []
              and aggregated.clarification_question == decision["clarification_question"])
        return ok, "clarification -> no execution"

    expected_failed = [a for a in order if a in fail or a in missing]
    expected_exec = [a for a in order if a not in fail and a not in missing]
    if not expected_failed:
        expected_status = "SUCCESS"
    elif expected_exec:
        expected_status = "PARTIAL_SUCCESS"
    else:
        expected_status = "FAILED"

    response_order = [r.agent_name for r in aggregated.responses]

    checks = {
        "executed_agents": aggregated.executed_agents == expected_exec,
        "failed_agents": aggregated.failed_agents == expected_failed,
        "response_count": len(aggregated.responses) == len(order),
        "response_order": response_order == order,  # strict execution order preserved
        "status": aggregated.status == expected_status,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return (not failed_checks), ("OK" if not failed_checks else f"failed: {failed_checks}")


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #

def _write_block(handle, entry, aggregated, elapsed, check_status) -> None:
    decision = entry["decision"]
    block = [
        SECTION,
        f"Benchmark ID    : {entry['id']}",
        f"Category        : {entry['category']}",
        f"User Query      : {entry['user_query']}",
        f"Planner Decision: {json.dumps(decision, ensure_ascii=False)}",
        f"Execution Order : {decision['execution_order']}",
        f"Executed Agents : {aggregated.executed_agents}",
        f"Failed Agents   : {aggregated.failed_agents}",
        f"Execution Status: {aggregated.status}",
        f"Execution Time  : {elapsed:.6f}s",
        "Agent Responses :",
        json.dumps([r.to_dict() for r in aggregated.responses], ensure_ascii=False, indent=2),
        "Aggregated Response:",
        json.dumps(aggregated.to_dict(), ensure_ascii=False, indent=2),
        f"Check           : {check_status}",
        SECTION,
        "",
    ]
    handle.write("\n".join(block) + "\n")
    handle.flush()


def _run_error_handling_checks(handle) -> tuple[int, int]:
    """Verify the deterministic pre-flight exceptions. Returns (passed, total)."""
    orchestrator = SupervisorOrchestrator(agents=_build_registry(set(), set()))
    cases = []

    # Duplicate agent.
    dup = PlannerDecision(
        intent=IntentType.MIXED_QUERY, confidence=ConfidenceLevel.HIGH,
        requires_clarification=False, clarification_question=None,
        agents=(AgentName.DATA_AGENT, AgentName.DATA_AGENT),
        execution_order=(AgentName.DATA_AGENT, AgentName.DATA_AGENT),
        reason="duplicate")
    cases.append(("DuplicateAgentError", dup, DuplicateAgentError))

    # Invalid execution order (not a permutation of agents).
    bad_order = PlannerDecision(
        intent=IntentType.MIXED_QUERY, confidence=ConfidenceLevel.HIGH,
        requires_clarification=False, clarification_question=None,
        agents=(AgentName.DATA_AGENT, AgentName.KNOWLEDGE_AGENT),
        execution_order=(AgentName.DATA_AGENT, AgentName.PREDICTION_AGENT),
        reason="bad order")
    cases.append(("InvalidExecutionOrderError", bad_order, InvalidExecutionOrderError))

    # Planner mismatch: not clarifying but no agents selected.
    mismatch = PlannerDecision(
        intent=IntentType.DATA_QUERY, confidence=ConfidenceLevel.HIGH,
        requires_clarification=False, clarification_question=None,
        agents=(), execution_order=(), reason="mismatch")
    cases.append(("PlannerMismatchError", mismatch, PlannerMismatchError))

    handle.write(f"{SECTION}\nERROR HANDLING CHECKS\n{SECTION}\n")
    passed = 0
    for name, decision, expected_error in cases:
        try:
            orchestrator.orchestrate(decision, "test", session_id="err")
            status = f"FAIL (no exception; expected {expected_error.__name__})"
        except expected_error:
            status = "PASS"
            passed += 1
        except Exception as error:  # noqa: BLE001
            status = f"FAIL (raised {type(error).__name__}, expected {expected_error.__name__})"
        handle.write(f"  [{status}] {name}\n")
    handle.write(SECTION + "\n\n")
    handle.flush()
    return passed, len(cases)


def main() -> int:
    entries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    total = len(entries)

    successful = 0
    failed = 0
    exec_times: list[float] = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"{SECTION}\nSUPERVISOR ORCHESTRATOR — INTEGRATION RESULTS\n{SECTION}\n\n")
        handle.flush()

        for entry in entries:
            decision = _build_decision(entry["decision"])
            registry = _build_registry(
                set(entry.get("fail_agents", [])), set(entry.get("missing_agents", []))
            )
            orchestrator = SupervisorOrchestrator(agents=registry)

            start = perf_counter()
            aggregated = orchestrator.orchestrate(
                decision,
                entry["user_query"],
                session_id=f"session-{entry['id']}",
                conversation_context=entry.get("memory"),
                request_id=f"req-{entry['id']}",
            )
            elapsed = perf_counter() - start
            exec_times.append(elapsed)

            passed, note = _evaluate(entry, aggregated)
            check_status = "PASS" if passed else f"FAIL ({note})"
            if passed:
                successful += 1
            else:
                failed += 1
                print(f"[{entry['id']}] ({entry['category']}) FAIL: {note}")

            _write_block(handle, entry, aggregated, elapsed, check_status)

        error_passed, error_total = _run_error_handling_checks(handle)

        average_time = (sum(exec_times) / len(exec_times)) if exec_times else 0.0
        overall_pass = (failed == 0 and successful == total
                        and error_passed == error_total)

        summary = [
            SECTION,
            "SUPERVISOR ORCHESTRATOR SUMMARY",
            SECTION,
            f"Total Queries        : {total}",
            f"Successful Executions: {successful}",
            f"Failed Executions    : {failed}",
            f"Error-Handling Checks: {error_passed}/{error_total}",
            f"Average Execution Time: {average_time:.6f}s",
            f"Overall Status       : {'PASS' if overall_pass else 'FAIL'}",
            SECTION,
        ]
        handle.write("\n".join(summary) + "\n")
        handle.flush()

    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Benchmarks: {successful}/{total} passed | Error-handling: {error_passed}/{error_total}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
