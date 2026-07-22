"""FINAL production end-to-end validation harness for AquaMind AI.

Runs a small, CURATED benchmark (``benchmark_pipeline.json``) through the
COMPLETE REAL production pipeline exactly as the future UI will, and writes a
human-readable regression report to ``production_pipeline_result.txt`` at the
project root:

    Benchmark Query  (stands in for future UI input)
      -> Conversation Memory   (real ConversationMemory component)
      -> Supervisor Planner    (real OpenCode Zen deepseek-v4-flash-free LLM)
      -> Planner Validator     (real PlannerValidator, inside Planner.plan)
      -> Validated PlannerDecision
      -> Supervisor Orchestrator
      -> Execution Engine
      -> Real specialist agents:
           * Data Agent       (SqlGenerator -> SQLiteExecutor -> EvidenceFormatter)
           * Knowledge Agent   (RetrievalCoordinator -> KnowledgeFormatter, real FAISS)
           * Prediction Agent  (trained XGBoost via PredictionAgentRuntime)
      -> UniversalAgentResponse(s)
      -> Response Aggregator
      -> AggregatedUniversalAgentResponse

The pipeline STOPS at the AggregatedUniversalAgentResponse. It does NOT invoke a
Recommendation Engine, Response Generator, or General LLM. Nothing is mocked.

The benchmark maximizes coverage, not quantity: ~25 curated cases spanning every
routing path and agent combination. The Prediction Agent has no natural-language
entry point, so its adapter reads slots (district / year / month / target) from
``AgentRequest.metadata`` -- NO NLP, NO regex.

=========================================================
PRODUCTION VALIDATION RULE
=========================================================
This harness validates the production pipeline; it never changes it. No
production component is modified to make a benchmark pass. If a benchmark
exposes a production defect, the defect is recorded, the failure is kept, and
the remaining benchmarks continue. The benchmark validates the system, it does
not adapt the system to itself.

=========================================================
PASS CRITERIA (per benchmark)
=========================================================
A benchmark PASSES only if:
  * the Planner JSON validated successfully (Planner.plan returned a decision),
  * for a clarification decision: the Orchestrator executed NO specialist agent,
  * otherwise:
      - execution order is respected (agents ran in the planner's order),
      - every selected real agent executed successfully,
      - the Response Aggregator completed and preserved every response exactly
        (same count, same order, evidence untouched),
      - the returned evidence is REAL (data = SQL + rows; knowledge = FAISS
        chunks with scores; prediction = a real model value) -- no mocks,
  * and no production component was bypassed (memory, planner, orchestrator and
    aggregator all ran).

The planner's routing is additionally compared against the benchmark ``expected``
block and reported as a diagnostic (routing match) without gating PASS.

Only transient free-tier LLM transport hiccups (empty / non-JSON / timeout /
connection) are retried; genuine defects (validation failures, real agent
errors, failed checks) are recorded immediately and never retried or masked.

Run:
    python agents/supervisor_agent/tests/production_pipeline_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

# --------------------------------------------------------------------------- #
# Locate project + production packages (import by location; no production edits)
# --------------------------------------------------------------------------- #

TEST_DIR = Path(__file__).resolve().parent
SUPERVISOR_DIR = TEST_DIR.parent
AGENTS_DIR = SUPERVISOR_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent

DATA_AGENT_DIR = AGENTS_DIR / "data_agent"
KNOWLEDGE_AGENT_DIR = AGENTS_DIR / "knowledge_agent"
PREDICTION_AGENT_DIR = AGENTS_DIR / "prediction_agent"

# Add the prediction agent's tests dir but NOT the data agent's tests dir; both
# contain a module named ``end_to_end_pipeline_test``. Keeping data/tests off
# sys.path makes reuse of the Prediction Agent runtime unambiguous.
_IMPORT_DIRS = (
    SUPERVISOR_DIR,                       # orchestrator / planner / memory packages
    DATA_AGENT_DIR / "llm",               # sql_generator
    DATA_AGENT_DIR / "database",          # sqlite_executor
    DATA_AGENT_DIR / "formatter",         # evidence_formatter
    KNOWLEDGE_AGENT_DIR,                  # knowledge_config
    KNOWLEDGE_AGENT_DIR / "retrieval",    # retrieval_coordinator
    KNOWLEDGE_AGENT_DIR / "formatter",    # knowledge_formatter
    PREDICTION_AGENT_DIR,                 # prediction_config
    PREDICTION_AGENT_DIR / "training",    # dataset_integrator, feature_engineering, model_registry
    PREDICTION_AGENT_DIR / "formatter",   # prediction_formatter
    PREDICTION_AGENT_DIR / "tests",       # prediction_benchmark_test (bench), PredictionAgentRuntime
)
for _dir in _IMPORT_DIRS:
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# --- supervisor packages --------------------------------------------------- #
from orchestrator import (  # noqa: E402
    AgentRequest,
    SupervisorOrchestrator,
    UniversalAgentResponse,
)
from planner.planner import Planner  # noqa: E402
from planner.planner_models import (  # noqa: E402
    PlannerApiError,
    PlannerResponseError,
    PlannerValidationError,
)
from memory import ConversationMemory  # noqa: E402

# --- real specialist components -------------------------------------------- #
from sql_generator import SqlGenerator  # noqa: E402  (Data Agent)
from sqlite_executor import SQLiteExecutor  # noqa: E402  (Data Agent)
from evidence_formatter import EvidenceFormatter  # noqa: E402  (Data Agent)
from retrieval_coordinator import RetrievalCoordinator  # noqa: E402  (Knowledge Agent)
from knowledge_formatter import KnowledgeFormatter  # noqa: E402  (Knowledge Agent)
from end_to_end_pipeline_test import PredictionAgentRuntime  # noqa: E402  (Prediction runtime, reused as-is)

BENCHMARK_PATH = TEST_DIR / "benchmark_pipeline.json"
OUTPUT_PATH = PROJECT_ROOT / "production_pipeline_result.txt"
SECTION = "=" * 60

# Bounded retry for transient free-tier LLM transport hiccups only.
MAX_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 3.0
_TRANSIENT_MARKERS = ("empty response", "timeout", "timed out", "connection",
                      "rate limit", "429", "temporarily")


# --------------------------------------------------------------------------- #
# Real specialist-agent adapters (SpecialistAgent: execute(request) -> response)
# --------------------------------------------------------------------------- #

class DataAgentAdapter:
    """Real Data Agent: SqlGenerator -> SQLiteExecutor -> EvidenceFormatter."""

    AGENT_NAME = "data_agent"

    def __init__(self) -> None:
        self._generator = SqlGenerator()
        self._executor = SQLiteExecutor()
        self._formatter = EvidenceFormatter()

    def execute(self, request: AgentRequest) -> UniversalAgentResponse:
        start = perf_counter()
        sql = self._generator.generate(request.user_query)
        rows = self._executor.execute(sql)
        evidence = self._formatter.format(rows)
        payload = {
            "agent_name": self.AGENT_NAME,
            "status": "SUCCESS" if evidence else "NO_RESULTS",
            "query_type": "data",
            "sql": sql,
            "row_count": len(evidence),
            "evidence": evidence,
        }
        return UniversalAgentResponse(self.AGENT_NAME, payload, perf_counter() - start)


class KnowledgeAgentAdapter:
    """Real Knowledge Agent: RetrievalCoordinator (FAISS) -> KnowledgeFormatter."""

    AGENT_NAME = "knowledge_agent"

    def __init__(self) -> None:
        self._coordinator = RetrievalCoordinator()
        self._formatter = KnowledgeFormatter()

    def execute(self, request: AgentRequest) -> UniversalAgentResponse:
        start = perf_counter()
        chunks = self._coordinator.retrieve(request.user_query)
        payload = self._formatter.format(chunks)
        return UniversalAgentResponse(self.AGENT_NAME, payload, perf_counter() - start)


class PredictionAgentAdapter:
    """Real Prediction Agent (trained XGBoost) driven by AgentRequest.metadata."""

    AGENT_NAME = "prediction_agent"

    def __init__(self) -> None:
        self._runtime = PredictionAgentRuntime()  # loads model + reference once

    def execute(self, request: AgentRequest) -> UniversalAgentResponse:
        start = perf_counter()
        metadata = request.metadata or {}
        district = metadata.get("district")
        year = metadata.get("prediction_year")
        if not district or year is None:
            raise ValueError(
                "Prediction Agent requires metadata 'district' and 'prediction_year'."
            )
        month = metadata.get("prediction_month")
        query = {"district": str(district), "year": int(year),
                 "month": int(month) if month else 1}
        outcome = self._runtime.run(query)
        return UniversalAgentResponse(self.AGENT_NAME, dict(outcome["response"]),
                                      perf_counter() - start)


# --------------------------------------------------------------------------- #
# Conversation Memory seeding (uses the REAL ConversationMemory component)
# --------------------------------------------------------------------------- #

def _prime_memory(memory: ConversationMemory, session_id: str, entry: dict) -> dict:
    """Create a clean session, replay prior turns/context, add the current turn.

    Returns the memory snapshot (Session.to_dict()) handed to the Planner.
    """
    if memory.session_exists(session_id):
        memory.delete_session(session_id)  # retries re-prime the same session id
    memory.create_session(session_id)
    cm = entry.get("conversation_memory") or {}

    for role, content in cm.get("history", []):
        if role == "user":
            memory.increment_turn(session_id)
            memory.add_user_message(session_id, content)
        else:
            memory.add_assistant_message(session_id, content)

    if cm.get("context"):
        memory.update_context(session_id, **cm["context"])

    memory.increment_turn(session_id)
    memory.add_user_message(session_id, entry["user_query"])
    return memory.get_session(session_id).to_dict()


# --------------------------------------------------------------------------- #
# Real-evidence authenticity (no mocks accepted)
# --------------------------------------------------------------------------- #

def _evidence_authentic(agent_name: str, payload: dict) -> bool:
    """True only for genuine production evidence from each real agent."""
    if agent_name == "data_agent":
        return (isinstance(payload.get("sql"), str) and bool(payload["sql"].strip())
                and isinstance(payload.get("evidence"), list))
    if agent_name == "knowledge_agent":
        evidence = payload.get("evidence")
        return (isinstance(evidence, list) and len(evidence) > 0
                and all(isinstance(item, dict) and "chunk_id" in item
                        and "similarity_score" in item and "document" in item
                        for item in evidence))
    if agent_name == "prediction_agent":
        prediction = payload.get("prediction")
        return (isinstance(prediction, dict)
                and isinstance(prediction.get("predicted_value"), (int, float))
                and bool(payload.get("model_name")))
    return False


def _is_transient_error(message: str | None) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _order_of(decision) -> list[str]:
    return [a.value if hasattr(a, "value") else str(a) for a in decision.execution_order]


# --------------------------------------------------------------------------- #
# Single attempt through the real pipeline
# --------------------------------------------------------------------------- #

def _run_once(entry: dict, memory: ConversationMemory, planner: Planner,
              orchestrator: SupervisorOrchestrator) -> dict:
    """Execute the full pipeline once for a benchmark entry; return a result dict.

    ``kind`` is one of: success, memory_fail, planner_transport, planner_validation,
    planner_error, orchestration_fail, clarification_bad, agent_transient,
    agent_defect, check_fail. ``transient`` marks retryable transport hiccups.
    """
    session_id = f"bench-{entry['id']}"
    result: dict[str, Any] = {
        "snapshot": None, "decision": None, "aggregated": None,
        "planner_seconds": 0.0, "orchestrate_seconds": 0.0, "aggregation_seconds": 0.0,
        "planner_validated": False, "checks": {}, "routing_match": None,
        "kind": "", "note": "", "transient": False,
    }
    expected = entry.get("expected", {})

    # --- Stage 1: real Conversation Memory ---
    try:
        result["snapshot"] = _prime_memory(memory, session_id, entry)
    except Exception as error:  # noqa: BLE001
        result.update(kind="memory_fail", transient=False,
                      note=f"FAIL (memory: {type(error).__name__}: {error})")
        return result

    # --- Stage 2: real Supervisor Planner LLM + Planner Validator (inside plan) ---
    try:
        t0 = perf_counter()
        decision = planner.plan(entry["user_query"], conversation_memory=result["snapshot"])
        result["planner_seconds"] = perf_counter() - t0
        result["decision"] = decision
        result["planner_validated"] = True
    except (PlannerApiError, PlannerResponseError) as error:
        result.update(kind="planner_transport", transient=True,
                      note=f"FAIL (planner transport: {type(error).__name__}: {error})")
        return result
    except PlannerValidationError as error:
        result.update(kind="planner_validation", transient=False,
                      note=f"FAIL (planner JSON invalid: {error})")
        return result
    except Exception as error:  # noqa: BLE001
        result.update(kind="planner_error", transient=False,
                      note=f"FAIL (planner: {type(error).__name__}: {error})")
        return result

    # Diagnostic: routing vs expected (never gates PASS).
    if decision.requires_clarification:
        result["routing_match"] = bool(expected.get("requires_clarification"))
    else:
        result["routing_match"] = (not expected.get("requires_clarification")
                                   and _order_of(decision) == list(expected.get("execution_order", [])))

    # --- Stages 3-5: Orchestrator -> Execution Engine -> real agents -> Aggregator ---
    try:
        t1 = perf_counter()
        aggregated = orchestrator.orchestrate(
            decision, entry["user_query"], session_id=session_id,
            conversation_context=result["snapshot"], request_id=f"prod-req-{entry['id']}",
            metadata=entry.get("metadata"),
        )
        result["orchestrate_seconds"] = perf_counter() - t1
        result["aggregated"] = aggregated
    except Exception as error:  # noqa: BLE001
        result.update(kind="orchestration_fail", transient=False,
                      note=f"FAIL (orchestration: {type(error).__name__}: {error})")
        return result

    result["aggregation_seconds"] = max(
        0.0, result["orchestrate_seconds"] - sum(r.execution_time for r in aggregated.responses))
    responses = aggregated.responses
    exec_order = _order_of(decision)

    # --- Clarification path: the orchestrator must not execute any agent ---
    if decision.requires_clarification:
        no_execution = (aggregated.status == "CLARIFICATION"
                        and aggregated.executed_agents == [] and responses == [])
        result["checks"] = {"clarification_no_execution": no_execution}
        if no_execution:
            result.update(kind="success", note="PASS (CLARIFICATION - no agents executed)")
        else:
            result.update(kind="clarification_bad", transient=False,
                          note="FAIL (clarification decision but orchestrator executed agents)")
        return result

    # --- Execution path: strengthened PASS checks ---
    if aggregated.status != "SUCCESS":
        transient = any(_is_transient_error(r.error) for r in responses if r.status == "ERROR")
        result.update(
            kind="agent_transient" if transient else "agent_defect", transient=transient,
            note=f"FAIL ({aggregated.status}; failed_agents={aggregated.failed_agents})")
        return result

    order_respected = [r.agent_name for r in responses] == exec_order
    all_succeeded = aggregated.failed_agents == [] and aggregated.status == "SUCCESS"
    evidence_authentic = all(_evidence_authentic(r.agent_name, r.payload) for r in responses)
    aggregator_preserved = (len(responses) == len(exec_order)
                            and [r.agent_name for r in responses] == exec_order
                            and all(r.to_dict() == r.payload for r in responses))
    result["checks"] = {
        "order_respected": order_respected,
        "all_agents_succeeded": all_succeeded,
        "evidence_authentic": evidence_authentic,
        "aggregator_preserved": aggregator_preserved,
    }

    if all(result["checks"].values()):
        result.update(kind="success", note="PASS (SUCCESS)")
    else:
        failed = [name for name, ok in result["checks"].items() if not ok]
        result.update(kind="check_fail", transient=False,
                      note=f"FAIL (checks failed: {failed})")
    return result


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #

def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_block(handle, entry: dict, result: dict, attempts: int) -> None:
    decision = result["decision"]
    aggregated = result["aggregated"]
    snapshot = result["snapshot"] or {}
    total_seconds = result["planner_seconds"] + result["orchestrate_seconds"]

    lines = [
        SECTION,
        f"Benchmark ID   : {entry['id']}",
        f"Category       : {entry['category']}",
        f"Difficulty     : {entry['difficulty']}",
        SECTION,
        f"User Query     : {entry['user_query']}",
        "Conversation Memory:",
        _dumps({"context": snapshot.get("context"), "history": snapshot.get("history")}),
        f"Planner JSON   : {'VALIDATED' if result['planner_validated'] else 'NOT VALIDATED'}",
        "Planner Decision:",
        _dumps(decision.to_dict()) if decision is not None else "N/A (planner failed)",
        f"Planner Time   : {result['planner_seconds']:.3f}s",
        f"Execution Order: {_order_of(decision) if decision is not None else []}",
        f"Executed Agents: {aggregated.executed_agents if aggregated is not None else []}",
        f"Routing Match  : {result['routing_match']}",
        SECTION,
    ]

    if aggregated is not None:
        for response in aggregated.responses:
            lines.extend([
                f"Agent Name     : {response.agent_name}",
                f"Execution Time : {response.execution_time:.3f}s",
                f"Status         : {response.status}",
                "Returned UniversalAgentResponse:",
                _dumps(response.to_dict()),
                SECTION,
            ])
        lines.extend([
            "Response Aggregator:",
            f"Status         : {aggregated.status}",
            f"Aggregation Time: {result['aggregation_seconds']:.3f}s",
            "AggregatedUniversalAgentResponse:",
            _dumps(aggregated.to_dict()),
            SECTION,
        ])

    if result["checks"]:
        lines.append(f"Validation Checks: {_dumps(result['checks'])}")
    passed = result["kind"] == "success"
    lines.extend([
        f"Total Pipeline Time: {total_seconds:.3f}s",
        f"Attempts       : {attempts}",
        f"{'PASS' if passed else 'FAIL'} - {result['note']}",
    ])
    if not passed:
        lines.append(f"Failure Reason : {result['note']}")
    lines.extend([SECTION, ""])

    handle.write("\n".join(lines) + "\n")
    handle.flush()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    entries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    total = len(entries)

    print("Initializing real production components (model + FAISS load)...")
    memory = ConversationMemory()
    planner = Planner()
    registry = {
        DataAgentAdapter.AGENT_NAME: DataAgentAdapter(),
        KnowledgeAgentAdapter.AGENT_NAME: KnowledgeAgentAdapter(),
        PredictionAgentAdapter.AGENT_NAME: PredictionAgentAdapter(),
    }
    orchestrator = SupervisorOrchestrator(agents=registry)
    print("Components ready. Running benchmark...\n")

    passed = failed = 0
    planner_failures = planner_validation_failures = memory_failures = 0
    orchestration_failures = data_failures = knowledge_failures = prediction_failures = 0
    clarification_requests = routing_mismatches = 0
    planner_times: list[float] = []
    agent_times: dict[str, list[float]] = {"data_agent": [], "knowledge_agent": [], "prediction_agent": []}
    aggregation_times: list[float] = []
    total_times: list[float] = []
    fastest: tuple[float, int] | None = None
    slowest: tuple[float, int] | None = None

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"{SECTION}\nAQUAMIND AI - PRODUCTION PIPELINE END-TO-END VALIDATION\n"
                     f"(real Conversation Memory, Planner+Validator, Orchestrator, Agents, Aggregator)\n"
                     f"{SECTION}\n\n")
        handle.flush()

        for entry in entries:
            result = _run_once(entry, memory, planner, orchestrator)
            attempts = 1
            # Retry ONLY transient transport hiccups; defects are kept as-is.
            while result["kind"] != "success" and result["transient"] and attempts < MAX_ATTEMPTS:
                print(f"[{entry['id']}] transient {result['note']} -> retry {attempts + 1}/{MAX_ATTEMPTS}")
                sleep(RETRY_PAUSE_SECONDS)
                result = _run_once(entry, memory, planner, orchestrator)
                attempts += 1

            decision = result["decision"]
            aggregated = result["aggregated"]

            if decision is not None:
                planner_times.append(result["planner_seconds"])
            if aggregated is not None:
                aggregation_times.append(result["aggregation_seconds"])
                for response in aggregated.responses:
                    if response.agent_name in agent_times:
                        agent_times[response.agent_name].append(response.execution_time)
                    if response.status == "ERROR":
                        if response.agent_name == "data_agent":
                            data_failures += 1
                        elif response.agent_name == "knowledge_agent":
                            knowledge_failures += 1
                        elif response.agent_name == "prediction_agent":
                            prediction_failures += 1
                if aggregated.status == "CLARIFICATION":
                    clarification_requests += 1

            if result["routing_match"] is False:
                routing_mismatches += 1

            total_seconds = result["planner_seconds"] + result["orchestrate_seconds"]
            kind = result["kind"]
            if kind == "success":
                passed += 1
                total_times.append(total_seconds)
                if fastest is None or total_seconds < fastest[0]:
                    fastest = (total_seconds, entry["id"])
                if slowest is None or total_seconds > slowest[0]:
                    slowest = (total_seconds, entry["id"])
            else:
                failed += 1
                if kind in ("planner_transport", "planner_error"):
                    planner_failures += 1
                elif kind == "planner_validation":
                    planner_validation_failures += 1
                elif kind == "memory_fail":
                    memory_failures += 1
                elif kind in ("orchestration_fail", "clarification_bad"):
                    orchestration_failures += 1
                print(f"[{entry['id']}] {result['note']}"
                      + (f" [attempts={attempts}]" if attempts > 1 else ""))

            _write_block(handle, entry, result, attempts)

        def _avg(values: list[float]) -> float:
            return (sum(values) / len(values)) if values else 0.0

        success_rate = (passed / total * 100.0) if total else 0.0
        overall_pass = (failed == 0 and passed == total)

        summary = [
            SECTION, "PIPELINE SUMMARY", SECTION,
            f"Total Benchmarks           : {total}",
            f"Passed                     : {passed}",
            f"Failed                     : {failed}",
            f"Planner Failures           : {planner_failures}",
            f"Planner Validation Failures: {planner_validation_failures}",
            f"Conversation Memory Failures: {memory_failures}",
            f"Data Agent Failures        : {data_failures}",
            f"Knowledge Agent Failures   : {knowledge_failures}",
            f"Prediction Agent Failures  : {prediction_failures}",
            f"Aggregation Failures       : {orchestration_failures}",
            f"Clarification Requests     : {clarification_requests}",
            f"Routing Mismatches (vs expected): {routing_mismatches}",
            f"Average Planner Time       : {_avg(planner_times):.3f}s",
            f"Average Data Agent Time    : {_avg(agent_times['data_agent']):.3f}s",
            f"Average Knowledge Agent Time: {_avg(agent_times['knowledge_agent']):.3f}s",
            f"Average Prediction Agent Time: {_avg(agent_times['prediction_agent']):.3f}s",
            f"Average Aggregation Time   : {_avg(aggregation_times):.3f}s",
            f"Average Total Pipeline Time: {_avg(total_times):.3f}s",
            f"Fastest Query              : " + (f"#{fastest[1]} ({fastest[0]:.3f}s)" if fastest else "n/a"),
            f"Slowest Query              : " + (f"#{slowest[1]} ({slowest[0]:.3f}s)" if slowest else "n/a"),
            f"Success Rate               : {success_rate:.1f}%",
            f"Overall Status             : {'PASS' if overall_pass else 'FAIL'}",
            SECTION,
        ]
        handle.write("\n".join(summary) + "\n")
        handle.flush()

    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Passed: {passed}/{total} | Planner: {planner_failures} Validation: {planner_validation_failures} | "
          f"Data: {data_failures} Knowledge: {knowledge_failures} Prediction: {prediction_failures} | "
          f"Routing mismatches: {routing_mismatches}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial results are in production_pipeline_result.txt")
        raise
