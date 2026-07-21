"""Production end-to-end integration test for the AquaMind AI Recommendation Layer.

Validates that the Recommendation Layer integrates correctly with the COMPLETE
real production pipeline. Nothing is mocked; every component is the real one.

    Benchmark Query  (stands in for future UI input)
      -> Conversation Memory   (real)
      -> Supervisor Planner + Planner Validator   (real DeepSeek V4 Flash)
      -> Supervisor Orchestrator + Execution Engine   (real)
      -> Real Data / Knowledge / Prediction agents
      -> Response Aggregator
      -> AggregatedUniversalAgentResponse
      -> Recommendation Decision   (real)  -> RecommendationDecision
      -> (if required) Recommendation Generator  (real) -> RecommendationResponse
    STOP.  (No Response Generator, no Conversation Memory update.)

The real pipeline wiring (agent adapters + Conversation Memory + Planner +
Orchestrator) is REUSED from the supervisor production pipeline harness rather
than duplicated. The 10 queries are chosen so recommendations SHOULD be
generated (over-exploited/declining/forecast-deterioration/actionable-mixed).

Results are flushed to ``recommendation_pipeline_result.txt`` after every case.

Run:
    python agents/recommendation_agent/tests/recommendation_pipeline_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter, sleep

TEST_DIR = Path(__file__).resolve().parent
RECOMMENDATION_DIR = TEST_DIR.parent
AGENTS_DIR = RECOMMENDATION_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
SUPERVISOR_TESTS_DIR = AGENTS_DIR / "supervisor_agent" / "tests"

for _dir in (AGENTS_DIR, SUPERVISOR_TESTS_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Reuse the real production pipeline wiring (adapters + memory seeding) as-is.
import production_pipeline_test as pipe  # noqa: E402
from recommendation_agent import (  # noqa: E402
    LlmClient,
    RecommendationDecider,
    RecommendationGenerator,
)

OUTPUT_PATH = PROJECT_ROOT / "recommendation_pipeline_result.txt"
SECTION = "=" * 60
MAX_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 3.0

# 10 production queries chosen so recommendations SHOULD be generated. Prediction
# involving cases carry the metadata slots the Prediction Agent needs.
BENCHMARKS = [
    {"id": 1, "user_query": "What is the stage of groundwater extraction in Chennai?", "metadata": {}},
    {"id": 2, "user_query": "How many over-exploited firkas are there in Tamil Nadu?", "metadata": {}},
    {"id": 3, "user_query": "Show the groundwater level in Erode and explain the causes of groundwater depletion.", "metadata": {}},
    {"id": 4, "user_query": "Predict the groundwater level in Chennai for 2031 and explain artificial recharge techniques.",
     "metadata": {"district": "Chennai", "prediction_year": 2031, "prediction_month": None, "target": "groundwater_level_m"}},
    {"id": 5, "user_query": "Predict the groundwater level in Salem for 2030, compare it with the current level, and explain recharge.",
     "metadata": {"district": "Salem", "prediction_year": 2030, "prediction_month": None, "target": "groundwater_level_m"}},
    {"id": 6, "user_query": "What is the current groundwater level in Coimbatore and explain why it is low?", "metadata": {}},
    {"id": 7, "user_query": "Compare the current groundwater level in Salem with the predicted level for 2030.",
     "metadata": {"district": "Salem", "prediction_year": 2030, "prediction_month": None, "target": "groundwater_level_m"}},
    {"id": 8, "user_query": "Predict the groundwater level in Madurai for 2029 and explain how communities can conserve water.",
     "metadata": {"district": "Madurai", "prediction_year": 2029, "prediction_month": None, "target": "groundwater_level_m"}},
    {"id": 9, "user_query": "Which five districts have the highest stage of groundwater extraction?", "metadata": {}},
    {"id": 10, "user_query": "Show the groundwater extraction stage in Karur and explain what over-exploitation means and how to address it.", "metadata": {}},
]


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _looks_transient(text: str) -> bool:
    text = (text or "").lower()
    return any(m in text for m in ("empty response", "timeout", "timed out", "connection",
                                   "rate limit", "429", "temporarily"))


def _run_case_once(entry, memory, planner, orchestrator, decider, generator) -> dict:
    """One full pipeline + recommendation pass; returns a structured result."""
    session_id = f"rec-bench-{entry['id']}"
    result = {"decision": None, "recommendation": None, "aggregated": None, "plan": None,
              "planner_seconds": 0.0, "agent_seconds": 0.0, "decision_seconds": 0.0,
              "generation_seconds": 0.0, "ok": False, "note": "", "transient": False}
    try:
        snapshot = pipe._prime_memory(memory, session_id, entry)
        t0 = perf_counter()
        decision_plan = planner.plan(entry["user_query"], conversation_memory=snapshot)
        result["planner_seconds"] = perf_counter() - t0
        result["plan"] = decision_plan

        t1 = perf_counter()
        aggregated = orchestrator.orchestrate(
            decision_plan, entry["user_query"], session_id=session_id,
            conversation_context=snapshot, request_id=f"rec-req-{entry['id']}",
            metadata=entry.get("metadata"))
        result["agent_seconds"] = perf_counter() - t1
        result["aggregated"] = aggregated

        if aggregated.status != "SUCCESS":
            transient = any(_looks_transient(r.error) for r in aggregated.responses if r.status == "ERROR")
            result.update(note=f"pipeline status {aggregated.status}; failed={aggregated.failed_agents}",
                          transient=transient)
            return result

        t2 = perf_counter()
        rec_decision = decider.decide(aggregated)
        result["decision_seconds"] = perf_counter() - t2
        result["decision"] = rec_decision

        if not rec_decision.recommendation_required:
            result.update(note="decision returned recommendation_required=false")
            return result

        t3 = perf_counter()
        recommendation = generator.generate(aggregated, rec_decision)
        result["generation_seconds"] = perf_counter() - t3
        result["recommendation"] = recommendation
        result.update(ok=True, note="PASS")
        return result
    except Exception as error:  # noqa: BLE001
        result.update(note=f"{type(error).__name__}: {error}", transient=_looks_transient(str(error)))
        return result


def _write_block(handle, entry, result, attempts) -> None:
    plan = result["plan"]
    aggregated = result["aggregated"]
    decision = result["decision"]
    recommendation = result["recommendation"]
    total = (result["planner_seconds"] + result["agent_seconds"]
             + result["decision_seconds"] + result["generation_seconds"])
    order = pipe._order_of(plan) if plan is not None else []
    lines = [
        SECTION,
        f"Benchmark ID   : {entry['id']}",
        f"User Query     : {entry['user_query']}",
        "Planner Decision:",
        _dumps(plan.to_dict()) if plan is not None else "N/A",
        f"Execution Order: {order}",
        f"Executed Agents: {aggregated.executed_agents if aggregated is not None else []}",
        "AggregatedUniversalAgentResponse:",
        _dumps(aggregated.to_dict()) if aggregated is not None else "N/A",
        "Recommendation Decision:",
        _dumps(decision.to_dict()) if decision is not None else "N/A",
        "Recommendation Response:",
        _dumps(recommendation.to_dict()) if recommendation is not None
        else ("(not required)" if decision is not None and not decision.recommendation_required else "N/A"),
        f"Planner Time   : {result['planner_seconds']:.3f}s",
        f"Agent Execution Time: {result['agent_seconds']:.3f}s",
        f"Recommendation Decision Time: {result['decision_seconds']:.3f}s",
        f"Recommendation Generation Time: {result['generation_seconds']:.3f}s",
        f"Pipeline Execution Time: {total:.3f}s",
        f"Attempts       : {attempts}",
        f"{'PASS' if result['ok'] else 'FAIL'} - {result['note']}",
        SECTION, "",
    ]
    handle.write("\n".join(lines) + "\n")
    handle.flush()


def main() -> int:
    total = len(BENCHMARKS)
    print("Initializing real production pipeline + Recommendation Layer...")
    memory = pipe.ConversationMemory()
    planner = pipe.Planner()
    registry = {
        pipe.DataAgentAdapter.AGENT_NAME: pipe.DataAgentAdapter(),
        pipe.KnowledgeAgentAdapter.AGENT_NAME: pipe.KnowledgeAgentAdapter(),
        pipe.PredictionAgentAdapter.AGENT_NAME: pipe.PredictionAgentAdapter(),
    }
    orchestrator = pipe.SupervisorOrchestrator(agents=registry)
    rec_client = LlmClient()
    decider = RecommendationDecider(client=rec_client)
    generator = RecommendationGenerator(client=rec_client)
    print("Ready. Running end-to-end recommendation pipeline...\n")

    passed = failed = 0
    recommendations_generated = recommendation_failures = 0
    planner_times: list[float] = []
    agent_times: list[float] = []
    decision_times: list[float] = []
    generation_times: list[float] = []
    total_times: list[float] = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"{SECTION}\nAQUAMIND AI - RECOMMENDATION LAYER END-TO-END PIPELINE VALIDATION\n"
                     f"(real Memory, Planner, Orchestrator, Agents, Aggregator, Recommendation Layer)\n{SECTION}\n\n")
        handle.flush()

        for entry in BENCHMARKS:
            result = _run_case_once(entry, memory, planner, orchestrator, decider, generator)
            attempts = 1
            while not result["ok"] and result["transient"] and attempts < MAX_ATTEMPTS:
                print(f"[{entry['id']}] transient ({result['note']}) -> retry {attempts + 1}/{MAX_ATTEMPTS}")
                sleep(RETRY_PAUSE_SECONDS)
                result = _run_case_once(entry, memory, planner, orchestrator, decider, generator)
                attempts += 1

            if result["plan"] is not None:
                planner_times.append(result["planner_seconds"])
            if result["aggregated"] is not None:
                agent_times.append(result["agent_seconds"])
            if result["decision"] is not None:
                decision_times.append(result["decision_seconds"])

            if result["ok"]:
                passed += 1
                recommendations_generated += 1
                generation_times.append(result["generation_seconds"])
                total_times.append(result["planner_seconds"] + result["agent_seconds"]
                                   + result["decision_seconds"] + result["generation_seconds"])
            else:
                failed += 1
                if result["decision"] is not None and result["decision"].recommendation_required:
                    recommendation_failures += 1
                print(f"[{entry['id']}] FAIL: {result['note']}"
                      + (f" [attempts={attempts}]" if attempts > 1 else ""))

            _write_block(handle, entry, result, attempts)

        def _avg(values: list[float]) -> float:
            return (sum(values) / len(values)) if values else 0.0

        overall_pass = (failed == 0 and passed == total)
        summary = [
            SECTION, "RECOMMENDATION PIPELINE SUMMARY", SECTION,
            f"Total Benchmarks              : {total}",
            f"Successful Pipeline Executions: {passed}",
            f"Failed Pipeline Executions    : {failed}",
            f"Recommendations Generated     : {recommendations_generated}",
            f"Recommendation Failures       : {recommendation_failures}",
            f"Average Planner Time          : {_avg(planner_times):.3f}s",
            f"Average Agent Execution Time  : {_avg(agent_times):.3f}s",
            f"Average Recommendation Decision Time: {_avg(decision_times):.3f}s",
            f"Average Recommendation Generation Time: {_avg(generation_times):.3f}s",
            f"Average Total Pipeline Time   : {_avg(total_times):.3f}s",
            f"Overall Status                : {'PASS' if overall_pass else 'FAIL'}",
            SECTION,
        ]
        handle.write("\n".join(summary) + "\n")
        handle.flush()

    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Passed: {passed}/{total} | Recommendations generated: {recommendations_generated} | "
          f"Recommendation failures: {recommendation_failures}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial results are in recommendation_pipeline_result.txt")
        raise
