"""Unit benchmark for the AquaMind AI Recommendation Layer.

Runs each curated AggregatedUniversalAgentResponse in ``recommendation_benchmark.json``
through the REAL Recommendation Decision, and -- only when the decision says so --
the REAL Recommendation Generator (both call DeepSeek V4 Flash at temperature 0).

    AggregatedUniversalAgentResponse
      -> Recommendation Decision  -> RecommendationDecision
      -> (if recommendation_required) Recommendation Generator -> RecommendationResponse

The benchmark AggregatedUniversalAgentResponses are curated fixtures (this is the
component's own unit benchmark). The separate ``recommendation_pipeline_test.py``
validates the Recommendation Layer against the REAL production pipeline.

Results are flushed to ``recommendation_result.txt`` after every case.

Run:
    python agents/recommendation_agent/tests/recommendation_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

TEST_DIR = Path(__file__).resolve().parent
RECOMMENDATION_DIR = TEST_DIR.parent
AGENTS_DIR = RECOMMENDATION_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from recommendation_agent import (  # noqa: E402
    LlmClient,
    RecommendationDecider,
    RecommendationGenerator,
)

BENCHMARK_PATH = TEST_DIR / "recommendation_benchmark.json"
OUTPUT_PATH = PROJECT_ROOT / "recommendation_result.txt"
SECTION = "=" * 60


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    total = len(cases)

    print("Initializing Recommendation Layer (shared LLM client)...")
    client = LlmClient()                       # shared -> one rate-limit budget
    decider = RecommendationDecider(client=client)
    generator = RecommendationGenerator(client=client)
    print("Ready. Running recommendation benchmark...\n")

    successful = failed = 0
    required_count = not_required_count = 0
    generation_success = generation_failure = 0
    expected_matches = 0
    decision_times: list[float] = []
    generation_times: list[float] = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"{SECTION}\nAQUAMIND AI - RECOMMENDATION LAYER BENCHMARK\n{SECTION}\n\n")
        handle.flush()

        for case in cases:
            aggregated = case["aggregated_response"]
            expected = case.get("expected_recommendation_required")
            decision = None
            recommendation = None
            decision_seconds = 0.0
            generation_seconds = 0.0
            case_ok = True
            note = ""

            try:
                t0 = perf_counter()
                decision = decider.decide(aggregated)
                decision_seconds = perf_counter() - t0
                decision_times.append(decision_seconds)

                if decision.recommendation_required:
                    required_count += 1
                    try:
                        t1 = perf_counter()
                        recommendation = generator.generate(aggregated, decision)
                        generation_seconds = perf_counter() - t1
                        generation_times.append(generation_seconds)
                        generation_success += 1
                    except Exception as error:  # noqa: BLE001
                        generation_failure += 1
                        case_ok = False
                        note = f"generation failed: {type(error).__name__}: {error}"
                else:
                    not_required_count += 1
            except Exception as error:  # noqa: BLE001
                case_ok = False
                note = f"decision failed: {type(error).__name__}: {error}"

            if decision is not None and expected is not None and decision.recommendation_required == expected:
                expected_matches += 1

            if case_ok:
                successful += 1
            else:
                failed += 1
                print(f"[{case['id']}] FAIL: {note}")

            block = [
                SECTION,
                f"Benchmark ID          : {case['id']}",
                f"Category              : {case['category']}",
                f"Description           : {case['description']}",
                f"Expected Required     : {expected}",
                "Recommendation Decision:",
                _dumps(decision.to_dict()) if decision is not None else f"N/A ({note})",
                "Recommendation Generated:",
                _dumps(recommendation.to_dict()) if recommendation is not None
                else ("(not required)" if decision is not None and not decision.recommendation_required
                      else f"N/A ({note})" if not case_ok else "(none)"),
                f"Decision Time         : {decision_seconds:.3f}s",
                f"Generation Time       : {generation_seconds:.3f}s",
                f"Execution Time        : {decision_seconds + generation_seconds:.3f}s",
                f"Result                : {'PASS' if case_ok else 'FAIL'}"
                + (f" ({note})" if note else ""),
                SECTION,
                "",
            ]
            handle.write("\n".join(block) + "\n")
            handle.flush()

        def _avg(values: list[float]) -> float:
            return (sum(values) / len(values)) if values else 0.0

        overall_pass = (failed == 0 and successful == total)
        summary = [
            SECTION, "RECOMMENDATION LAYER SUMMARY", SECTION,
            f"Total Cases               : {total}",
            f"Recommendation Required   : {required_count}",
            f"Recommendation Not Required: {not_required_count}",
            f"Generation Success        : {generation_success}",
            f"Generation Failure        : {generation_failure}",
            f"Decision Matched Expected : {expected_matches}/{total}",
            f"Average Decision Time     : {_avg(decision_times):.3f}s",
            f"Average Generation Time   : {_avg(generation_times):.3f}s",
            f"Overall Status            : {'PASS' if overall_pass else 'FAIL'}",
            SECTION,
        ]
        handle.write("\n".join(summary) + "\n")
        handle.flush()

    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Passed: {successful}/{total} | Required: {required_count} | "
          f"Gen success: {generation_success} | Gen fail: {generation_failure} | "
          f"Decision matched expected: {expected_matches}/{total}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
