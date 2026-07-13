"""Temporary integration test for the AquaMind AI Supervisor Planner.

NOT production code. Runs every benchmark query through the real planning
pipeline and writes a report to ``plannerllmresult.txt`` at the project root:

    planner_benchmark.json
      -> Conversation Memory (built for follow-up entries)
      -> PromptBuilder (assemble planning prompt)
      -> Planner LLM (deepseek-v4-flash-free, temperature 0)
      -> PlannerValidator (strict routing-JSON validation)
      -> validated PlannerDecision

It does NOT execute any specialist agent. A benchmark passes when the planner
returns a routing decision that validates successfully. Failures are recorded
and the run continues.

Run:
    python agents/supervisor_agent/tests/planner_test.py
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

from memory import ConversationMemory  # noqa: E402
from planner import Planner, PlannerValidationError  # noqa: E402
from planner.planner_models import (  # noqa: E402
    EmptyQueryError,
    PlannerApiError,
    PlannerResponseError,
)

BENCHMARK_PATH = SUPERVISOR_DIR / "planner_benchmark.json"
OUTPUT_PATH = PROJECT_ROOT / "plannerllmresult.txt"
SECTION = "=" * 50


def _memory_snapshot(entry: dict) -> dict | None:
    """Build a Conversation Memory session for a follow-up entry, then snapshot it.

    Uses the real ConversationMemory API. Returns None when the entry carries no
    prior context.
    """
    if "memory" not in entry:
        return None
    memory = ConversationMemory()
    session_id = f"benchmark-{entry['id']}"
    memory.create_session(session_id)
    for role, content in entry.get("messages", []):
        if role == "user":
            memory.add_user_message(session_id, content)
        else:
            memory.add_assistant_message(session_id, content)
    memory.update_context(session_id, **entry["memory"])
    return memory.get_session(session_id).to_dict()


def _write_block(handle, *, benchmark_id, category, query, snapshot, prompt_length,
                 elapsed, validation_status, raw_json, decision_text) -> None:
    """Append one benchmark result to the report and flush it to disk immediately."""
    block = [
        SECTION,
        f"Benchmark ID   : {benchmark_id}",
        f"Category       : {category}",
        f"User Query     : {query}",
    ]
    if snapshot is not None:
        block.append("Conversation Memory: provided (follow-up)")
    block += [
        f"Prompt Length  : {prompt_length} chars",
        f"Execution Time : {elapsed:.3f}s",
        f"Validation Status: {validation_status}",
        "Planner Raw JSON:",
        raw_json or "(empty)",
        "Validated Planner Decision:",
        decision_text,
        SECTION,
        "",
    ]
    handle.write("\n".join(block) + "\n")
    handle.flush()  # persist immediately so nothing is lost if the run is interrupted


def main() -> int:
    queries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    total = len(queries)
    planner = Planner()

    processed = 0
    successful = 0
    validation_failures = 0
    api_failures = 0
    planner_times: list[float] = []
    interrupted = False
    overall_pass = False

    # Open the report up front and write results incrementally. Even if the run
    # is interrupted (API death, Ctrl-C), every completed result is already on disk
    # and a summary reflecting progress-so-far is written in the finally block.
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"{SECTION}\nSUPERVISOR PLANNER BENCHMARK — INCREMENTAL RESULTS\n{SECTION}\n\n")
        handle.flush()
        try:
            for entry in queries:
                benchmark_id = entry["id"]
                category = entry["category"]
                query = entry["query"]
                snapshot = _memory_snapshot(entry)

                print(f"[{benchmark_id}/{total}] ({category}) {query}")

                start = perf_counter()
                try:
                    decision = planner.plan(query, snapshot)
                    elapsed = perf_counter() - start
                    raw_json = planner.last_raw_response.strip()
                    decision_text = json.dumps(decision.to_dict(), ensure_ascii=False, indent=2)
                    validation_status = "VALID"
                    successful += 1
                    planner_times.append(elapsed)
                except PlannerValidationError as error:
                    elapsed = perf_counter() - start
                    raw_json = (planner.last_raw_response or "").strip()
                    decision_text = "(validation failed)"
                    validation_status = f"VALIDATION_FAILED: {error}"
                    validation_failures += 1
                    planner_times.append(elapsed)
                    print(f"    VALIDATION_FAILED: {error}")
                except (PlannerApiError, PlannerResponseError, EmptyQueryError) as error:
                    elapsed = perf_counter() - start
                    raw_json = (planner.last_raw_response or "").strip()
                    decision_text = "(no decision)"
                    validation_status = f"API_FAILURE: {type(error).__name__}: {error}"
                    api_failures += 1
                    print(f"    API_FAILURE: {type(error).__name__}: {error}")

                processed += 1
                _write_block(
                    handle,
                    benchmark_id=benchmark_id,
                    category=category,
                    query=query,
                    snapshot=snapshot,
                    prompt_length=planner.last_prompt_length,
                    elapsed=elapsed,
                    validation_status=validation_status,
                    raw_json=raw_json,
                    decision_text=decision_text,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted — writing partial summary.")
        finally:
            average_time = (sum(planner_times) / len(planner_times)) if planner_times else 0.0
            complete = (processed == total) and not interrupted
            overall_pass = complete and validation_failures == 0 and api_failures == 0

            summary = [
                SECTION,
                "SUPERVISOR PLANNER BENCHMARK SUMMARY",
                SECTION,
                f"Total Queries        : {total}",
                f"Processed Queries    : {processed}",
                f"Successful Planning  : {successful}",
                f"Validation Failures  : {validation_failures}",
                f"API Failures         : {api_failures}",
                f"Average Planner Time : {average_time:.3f}s",
                f"Rate Limit Hits      : {planner.rate_limit_hits}",
                f"Run Completed        : {complete}",
                f"Overall Status       : {'PASS' if overall_pass else 'FAIL'}",
                SECTION,
            ]
            handle.write("\n".join(summary) + "\n")
            handle.flush()

    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Processed: {processed}/{total} | Successful: {successful} | "
          f"Validation failures: {validation_failures} | API failures: {api_failures}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
