"""End-to-end benchmark harness for the AquaMind AI Response Generator.

This harness plays the role of the Supervisor Orchestrator. For each benchmark
case it assembles the complete prompt in the deterministic order the production
orchestrator uses:

    1. response_system_prompt.txt
    2. User Query
    3. Updated AggregatedUniversalAgentResponse (sanitized: status + responses,
       with RecommendationResponse embedded ONLY when it was generated)

It then invokes the real, deterministic Response Generator (DeepSeek V4 Flash,
temperature 0) and validates that the returned FinalResponse is grounded strictly
in the supplied evidence and leaks no internal implementation detail.

The harness never modifies any production component. It only reads the benchmark
fixtures, assembles prompts, calls the generator, and writes an incremental
report to ``response_generator_result.txt`` at the project root.

Run:  python -u agents/response_generator/tests/response_generator_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

# --------------------------------------------------------------------------- #
# Path bootstrap: put the `agents/` directory on sys.path so both
# `response_generator` and its reused `recommendation_agent.config` import.
# --------------------------------------------------------------------------- #

THIS_FILE = Path(__file__).resolve()
TESTS_DIR = THIS_FILE.parent
RESPONSE_GENERATOR_DIR = TESTS_DIR.parent
AGENTS_DIR = RESPONSE_GENERATOR_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent

if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from response_generator.config import (  # noqa: E402
    LlmClient,
    RESPONSE_SYSTEM_PROMPT_PATH,
    load_prompt,
)
from response_generator.response_generator import ResponseGenerator  # noqa: E402
from response_generator.response_models import (  # noqa: E402
    FinalResponse,
    FinalResponseValidationError,
)

BENCHMARK_PATH = TESTS_DIR / "response_generator_benchmark.json"
RESULT_PATH = PROJECT_ROOT / "response_generator_result.txt"

# Terms that must never appear in a user-facing response (internal plumbing).
INTERNAL_LEAK_TERMS = (
    "planner_decision",
    "executed_agents",
    "failed_agents",
    "request_id",
    "session_id",
    "execution_context",
    "agentrequest",
    "conversation memory",
)


# --------------------------------------------------------------------------- #
# Prompt assembly (the Supervisor Orchestrator's responsibility, reproduced
# here for the benchmark ONLY).
# --------------------------------------------------------------------------- #

def build_sanitized_aggregate(
    aggregated_response: dict[str, Any],
    recommendation_response: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the Updated AggregatedUniversalAgentResponse the generator may see.

    Deliberately restricted to ``status`` and specialist ``responses``. When a
    RecommendationResponse was generated upstream, it is embedded here (exactly
    as produced) so the aggregate is "updated" as in production. Planner,
    routing, memory and identifier fields are never included.
    """
    sanitized: dict[str, Any] = {
        "status": aggregated_response.get("status"),
        "responses": aggregated_response.get("responses", []),
    }
    if recommendation_response is not None:
        sanitized["recommendation_response"] = recommendation_response
    return sanitized


def assemble_prompt(
    system_prompt: str,
    user_query: str,
    aggregated_response: dict[str, Any],
    recommendation_response: dict[str, Any] | None,
) -> str:
    """Assemble the complete prompt in the fixed production order."""
    sanitized = build_sanitized_aggregate(aggregated_response, recommendation_response)
    aggregate_json = json.dumps(sanitized, indent=2, ensure_ascii=False)
    return (
        f"{system_prompt}\n\n"
        "=========================================================\n"
        "USER QUERY\n"
        "=========================================================\n"
        f"{user_query}\n\n"
        "=========================================================\n"
        "UPDATED AGGREGATED UNIVERSAL AGENT RESPONSE\n"
        "=========================================================\n"
        f"{aggregate_json}\n"
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_response(case: dict[str, Any], result: FinalResponse) -> list[str]:
    """Return a list of failure reasons; empty means the case passed."""
    failures: list[str] = []

    if result.status != "SUCCESS":
        failures.append(f"status must be 'SUCCESS', got '{result.status}'")
    if not result.response.strip():
        failures.append("response is empty")

    lowered = result.response.lower()

    # An expected_terms entry may be a string (must appear) or a list of
    # alternatives (at least one must appear). The list form allows grounding to
    # be validated without pinning the model to one exact phrasing.
    for expected in case.get("expected_terms", []):
        if isinstance(expected, list):
            if not any(alt.lower() in lowered for alt in expected):
                failures.append(f"missing expected term (any of): {expected}")
        elif expected.lower() not in lowered:
            failures.append(f"missing expected term: '{expected}'")

    for term in case.get("forbidden_terms", []):
        if term.lower() in lowered:
            failures.append(f"contains forbidden term: '{term}'")

    for term in INTERNAL_LEAK_TERMS:
        if term in lowered:
            failures.append(f"leaked internal term: '{term}'")

    return failures


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_block(handle: TextIO, block: str) -> None:
    handle.write(block)
    handle.flush()


def format_case_block(
    case: dict[str, Any],
    used_recommendation: bool,
    response_text: str,
    elapsed: float,
    failures: list[str],
) -> str:
    verdict = "PASS" if not failures else "FAIL"
    lines = [
        "=" * 70,
        f"Benchmark ID     : {case.get('id')}",
        f"Category         : {case.get('category')}",
        f"User Query       : {case.get('user_query')}",
        f"Recommendation   : {'Yes' if used_recommendation else 'No'}",
        f"Execution Time   : {elapsed:.3f}s",
        f"Result           : {verdict}",
    ]
    if failures:
        lines.append("Failure Reasons  :")
        lines.extend(f"    - {reason}" for reason in failures)
    lines.append("Generated Response:")
    lines.append(response_text if response_text else "    <no response>")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = benchmark["cases"]
    system_prompt = load_prompt(RESPONSE_SYSTEM_PROMPT_PATH)

    client = LlmClient()
    generator = ResponseGenerator(client)

    total = len(cases)
    successes = 0
    failures_count = 0
    durations: list[float] = []

    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        write_block(
            handle,
            "AquaMind AI - Response Generator Benchmark Report\n"
            f"Model: {client.model_name}  (temperature=0, deterministic)\n"
            f"Total cases: {total}\n\n",
        )

        for case in cases:
            recommendation = case.get("recommendation_response")
            used_recommendation = recommendation is not None
            prompt = assemble_prompt(
                system_prompt,
                case["user_query"],
                case["aggregated_response"],
                recommendation,
            )

            start = time.monotonic()
            response_text = ""
            case_failures: list[str] = []
            try:
                result = generator.generate(prompt)
                elapsed = time.monotonic() - start
                response_text = result.response
                case_failures = validate_response(case, result)
            except (FinalResponseValidationError, Exception) as error:  # noqa: BLE001
                elapsed = time.monotonic() - start
                case_failures = [f"generation error: {type(error).__name__}: {error}"]

            durations.append(elapsed)
            if case_failures:
                failures_count += 1
            else:
                successes += 1

            block = format_case_block(
                case, used_recommendation, response_text, elapsed, case_failures
            )
            write_block(handle, block)
            print(f"[{case.get('id')}] {'PASS' if not case_failures else 'FAIL'} ({elapsed:.2f}s)")

        avg = sum(durations) / len(durations) if durations else 0.0
        overall = "PASS" if failures_count == 0 else "FAIL"
        summary = (
            "=" * 70 + "\n"
            "SUMMARY\n"
            + "=" * 70 + "\n"
            f"Total Cases          : {total}\n"
            f"Successful Responses : {successes}\n"
            f"Failed Responses     : {failures_count}\n"
            f"Average Generation Time : {avg:.3f}s\n"
            f"Overall Status       : {overall}\n"
        )
        write_block(handle, summary)
        print(summary)

    return 0 if failures_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
