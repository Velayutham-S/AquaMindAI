"""Temporary integration test for the AquaMind AI Knowledge Formatter.

NOT production code. Runs every benchmark question through the pipeline:

    benchmark_questions.json
      -> RetrievalCoordinator.retrieve()   (list[RetrievedChunk])
      -> KnowledgeFormatter.format()        (structured knowledge response)

and writes a human-readable report to ``knowledge_formatter_output.txt`` at the
project root for manual review. It calls no LLM, summarizes nothing, and modifies
no production files or chunk text.

Run:
    python agents/knowledge_agent/tests/knowledge_formatter_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

TEST_DIR = Path(__file__).resolve().parent
KNOWLEDGE_AGENT_DIR = TEST_DIR.parent
PROJECT_ROOT = KNOWLEDGE_AGENT_DIR.parents[1]
sys.path.insert(0, str(KNOWLEDGE_AGENT_DIR / "retrieval"))
sys.path.insert(0, str(KNOWLEDGE_AGENT_DIR / "formatter"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from retrieval_coordinator import RetrievalCoordinator  # noqa: E402
from knowledge_formatter import KnowledgeFormatter  # noqa: E402

BENCHMARK_PATH = KNOWLEDGE_AGENT_DIR / "retrieval" / "benchmark_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "knowledge_formatter_output.txt"
SECTION = "=" * 60
RULE = "-" * 60

EXPECTED_KEYS = {
    "agent_name", "status", "query_type",
    "total_evidence", "retrieval_method", "evidence",
}


def _is_valid_response(response: dict, chunk_count: int) -> bool:
    """A response is valid when it has the expected envelope and matching counts."""
    if not EXPECTED_KEYS.issubset(response.keys()):
        return False
    if response["agent_name"] != "knowledge_agent":
        return False
    if response["query_type"] != "knowledge":
        return False
    if response["retrieval_method"] != "semantic_search":
        return False
    if response["total_evidence"] != chunk_count:
        return False
    if len(response["evidence"]) != chunk_count:
        return False
    expected_status = "SUCCESS" if chunk_count else "NO_RESULTS"
    return response["status"] == expected_status


def main() -> int:
    questions = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    coordinator = RetrievalCoordinator()
    formatter = KnowledgeFormatter()

    lines: list[str] = []
    successful = 0
    failed = 0
    format_times: list[float] = []

    for item in questions:
        query_id = item.get("id")
        category = item.get("category", "?")
        difficulty = item.get("difficulty", "?")
        query = item.get("query", "")
        print(f"Running Benchmark ID {query_id}...")

        lines.append(SECTION)
        lines.append(f"Benchmark ID : {query_id}")
        lines.append(f"Category     : {category}")
        lines.append(f"Difficulty   : {difficulty}")
        lines.append(RULE)
        lines.append("USER QUERY")
        lines.append(query)
        lines.append(RULE)

        try:
            chunks = coordinator.retrieve(query)

            start = perf_counter()
            response = formatter.format(chunks)
            format_times.append(perf_counter() - start)

            lines.append(f"RETRIEVED CHUNKS ({len(chunks)})")
            if not chunks:
                lines.append("(no chunks)")
            for chunk in chunks:
                lines.append("")
                lines.append(f"  Rank       : {chunk.rank}")
                lines.append(f"  Document   : {chunk.document}")
                lines.append(f"  Page       : {chunk.page}")
                lines.append(f"  Score      : {chunk.score:.4f}")
                lines.append("  Chunk Text :")
                lines.append(chunk.text)
            lines.append(RULE)

            if _is_valid_response(response, len(chunks)):
                successful += 1
                print(f"OK Benchmark ID {query_id} formatted ({response['status']}).")
            else:
                failed += 1
                print(f"FAIL Benchmark ID {query_id}: invalid response envelope.")

            lines.append("FORMATTED KNOWLEDGE RESPONSE (UniversalAgentResponse)")
            lines.append(json.dumps(response, indent=2, ensure_ascii=False, default=str))
        except Exception as error:  # noqa: BLE001 - record per query, continue
            failed += 1
            lines.append(f"ERROR: {type(error).__name__}: {error}")
            print(f"FAIL Benchmark ID {query_id}: {type(error).__name__}: {error}")

        lines.append(SECTION)
        lines.append("")

    average_format_time = (sum(format_times) / len(format_times)) if format_times else 0.0
    overall_pass = failed == 0 and successful == len(questions)

    lines.append(SECTION)
    lines.append("KNOWLEDGE FORMATTER SUMMARY")
    lines.append(SECTION)
    lines.append(f"Total Queries          : {len(questions)}")
    lines.append(f"Successful Formatting   : {successful}")
    lines.append(f"Failed Formatting       : {failed}")
    lines.append(f"Average Formatting Time : {average_format_time:.6f}s")
    lines.append(f"Overall Status          : {'PASS' if overall_pass else 'FAIL'}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {OUTPUT_PATH}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'} "
          f"({successful}/{len(questions)} successful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
