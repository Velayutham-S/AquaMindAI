"""Temporary integration test for the AquaMind AI retrieval pipeline.

NOT production code. Runs every benchmark question through
``RetrievalCoordinator.retrieve()`` and writes a human-readable report to
``retrieve.txt`` at the project root so retrieval quality can be reviewed
before building the Knowledge Formatter.

It performs no LLM calls, no summarization, and does not alter retrieved text
or metadata.

Run:
    python agents/knowledge_agent/tests/retrieval_pipeline_test.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from retrieval_coordinator import RetrievalCoordinator  # noqa: E402

BENCHMARK_PATH = KNOWLEDGE_AGENT_DIR / "retrieval" / "benchmark_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "retrieve.txt"
SECTION = "=" * 60
RULE = "-" * 60


def main() -> int:
    questions = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    coordinator = RetrievalCoordinator()
    top_k = coordinator._config.top_k

    lines: list[str] = []
    successful = 0
    failed = 0
    times: list[float] = []

    for item in questions:
        query_id = item.get("id")
        category = item.get("category", "?")
        difficulty = item.get("difficulty", "?")
        query = item.get("query", "")

        lines.append(SECTION)
        lines.append("TEST CASE")
        lines.append(SECTION)
        lines.append(f"Benchmark ID : {query_id}")
        lines.append(f"Category     : {category}")
        lines.append(f"Difficulty   : {difficulty}")
        lines.append(RULE)
        lines.append("USER QUERY")
        lines.append(query)
        lines.append(RULE)
        lines.append(f"TOP-K RESULTS (k={top_k})")

        try:
            start = perf_counter()
            chunks = coordinator.retrieve(query)
            times.append(perf_counter() - start)
            successful += 1

            if not chunks:
                lines.append("(no results)")
            for chunk in chunks:
                lines.append("")
                lines.append(f"Rank             : {chunk.rank}")
                lines.append(f"Similarity Score : {chunk.score:.4f}")
                lines.append(f"Document         : {chunk.document}")
                lines.append(f"Category         : {chunk.category}")
                lines.append(f"Page             : {chunk.page}")
                lines.append(f"Section          : {chunk.section}")
                lines.append(f"Chunk ID         : {chunk.chunk_id}")
                lines.append(f"Source Path      : {chunk.source_path}")
                lines.append("Chunk Text       :")
                lines.append(chunk.text)
                lines.append(RULE)
        except Exception as error:  # noqa: BLE001 - record per query, continue
            failed += 1
            lines.append(f"ERROR: {type(error).__name__}: {error}")
            lines.append(RULE)
        lines.append("")

    average_time = (sum(times) / len(times)) if times else 0.0
    overall_pass = failed == 0 and successful == len(questions)

    lines.append(SECTION)
    lines.append("RETRIEVAL SUMMARY")
    lines.append(SECTION)
    lines.append(f"Total Queries               : {len(questions)}")
    lines.append(f"Top-K                       : {top_k}")
    lines.append(f"Embedding Model             : {coordinator.embedding_model}")
    lines.append(f"Embedding Dimension         : {coordinator.embedding_dimension}")
    lines.append(f"Vector Count                : {coordinator.vector_count}")
    lines.append(f"Total Successful Retrievals : {successful}")
    lines.append(f"Failed Retrievals           : {failed}")
    lines.append(f"Average Retrieval Time      : {average_time:.3f}s")
    lines.append(f"Overall Status              : {'PASS' if overall_pass else 'FAIL'}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {OUTPUT_PATH}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'} "
          f"({successful}/{len(questions)} successful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
