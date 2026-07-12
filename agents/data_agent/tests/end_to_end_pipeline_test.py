"""Temporary end-to-end integration test for the AquaMind AI Data Agent.

This script is NOT part of the production architecture. It exercises the full
pipeline for the first 10 benchmark queries and writes a report to
``endtest.txt`` in the project root:

    User Query
      -> SqlGenerator.generate()      (SQL generation + validation)
      -> SQLiteExecutor.execute()     (raw sqlite3.Row results)
      -> EvidenceFormatter.format()   (structured evidence)

It calls no Response Generator, summarizes nothing, and modifies no production
files. Displayed database rows and formatted evidence are capped at the first 5.

Run:
    python agents/data_agent/tests/end_to_end_pipeline_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

# --- locate project + production packages (no production files are imported by path hacks) ---
TEST_DIR = Path(__file__).resolve().parent
DATA_AGENT_DIR = TEST_DIR.parent
PROJECT_ROOT = DATA_AGENT_DIR.parents[1]  # data_agent -> agents -> root

sys.path.insert(0, str(DATA_AGENT_DIR / "llm"))
sys.path.insert(0, str(DATA_AGENT_DIR / "database"))
sys.path.insert(0, str(DATA_AGENT_DIR / "formatter"))

from sql_generator import SqlGenerator, ValidationError, SqlGeneratorError  # noqa: E402
from sqlite_executor import SQLiteExecutor, SQLiteExecutorError  # noqa: E402
from evidence_formatter import EvidenceFormatter  # noqa: E402

BENCHMARK_PATH = DATA_AGENT_DIR / "llm" / "llm_inputs" / "benchmark_queries.json"
OUTPUT_PATH = PROJECT_ROOT / "endtest.txt"
MAX_DISPLAY = 5
SECTION = "=" * 60
RULE = "-" * 60


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def main() -> None:
    queries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))[10:]

    generator = SqlGenerator()
    executor = SQLiteExecutor()
    formatter = EvidenceFormatter()

    lines: list[str] = []
    stats = {
        "total": len(queries),
        "gen_ok": 0, "gen_fail": 0,
        "exec_ok": 0, "exec_fail": 0,
        "fmt_ok": 0, "fmt_fail": 0,
        "gen_times": [], "exec_times": [], "fmt_times": [], "total_times": [],
    }

    for index, item in enumerate(queries, start=11):
        query_id = item.get("id")
        category = item.get("category", "?")
        difficulty = item.get("difficulty", "?")
        user_query = item.get("query", "")
        print(f"Running Benchmark ID {query_id}...")
        sql: str | None = None
        validation = "FAIL"
        rows: list = []
        evidence: list = []
        gen_time = exec_time = fmt_time = None
        notes: list[str] = []
        
        # --- Stage 1: SQL generation (includes validation inside generate) ---
        start = perf_counter()
        try:
            sql = generator.generate(user_query)
            gen_time = perf_counter() - start
            validation = "PASS"
            stats["gen_ok"] += 1
            print(f"✓ SQL generated for Benchmark ID {query_id}")
        except ValidationError as error:
            gen_time = perf_counter() - start
            sql = error.sql
            validation = "FAIL"
            stats["gen_fail"] += 1
            notes.append(f"validation rejected: {error}")
        except SqlGeneratorError as error:
            gen_time = perf_counter() - start
            stats["gen_fail"] += 1
            notes.append(f"generation error: {type(error).__name__}: {error}")

        # --- Stage 2: SQL execution (only when we have valid SQL) ---
        if validation == "PASS" and sql:
            start = perf_counter()
            try:
                rows = executor.execute(sql)
                exec_time = perf_counter() - start
                stats["exec_ok"] += 1
                print(f"✓ SQL executed for Benchmark ID {query_id}")
            except SQLiteExecutorError as error:
                exec_time = perf_counter() - start
                stats["exec_fail"] += 1
                notes.append(f"execution error: {type(error).__name__}: {error}")

        # --- Stage 3: Evidence formatting (only when execution succeeded) ---
        if exec_time is not None and "execution error" not in " ".join(notes):
            start = perf_counter()
            try:
                evidence = formatter.format(rows)
                fmt_time = perf_counter() - start
                stats["fmt_ok"] += 1
                print(f"✓ Evidence formatted for Benchmark ID {query_id}")
            except Exception as error:  # noqa: BLE001
                fmt_time = perf_counter() - start
                stats["fmt_fail"] += 1
                notes.append(f"formatting error: {type(error).__name__}: {error}")

        total_time = sum(t for t in (gen_time, exec_time, fmt_time) if t is not None)
        stats["gen_times"].append(gen_time if gen_time is not None else 0.0)
        if exec_time is not None:
            stats["exec_times"].append(exec_time)
        if fmt_time is not None:
            stats["fmt_times"].append(fmt_time)
        stats["total_times"].append(total_time)
        
        # --- write report block for this query ---
        lines.append(SECTION)
        lines.append(f"TEST CASE {index}")
        lines.append(SECTION)
        lines.append(f"Benchmark ID : {query_id}")
        lines.append(f"Category     : {category}")
        lines.append(f"Difficulty   : {difficulty}")
        lines.append(RULE)
        lines.append("USER QUERY")
        lines.append(user_query)
        lines.append(RULE)
        lines.append("GENERATED SQL")
        lines.append(sql if sql else "(no SQL generated)")
        lines.append(RULE)
        lines.append("SQL VALIDATION")
        lines.append(validation)
        if notes:
            for note in notes:
                lines.append(f"  note: {note}")
        lines.append(RULE)
        lines.append("ROWS RETURNED")
        lines.append(str(len(rows)))
        lines.append(RULE)
        lines.append(f"DATABASE ROW SAMPLE (first {MAX_DISPLAY})")
        if rows:
            lines.append(f"columns: {list(rows[0].keys())}")
            for row in rows[:MAX_DISPLAY]:
                lines.append(f"  {tuple(row)}")
        else:
            lines.append("(no rows)")
        lines.append(RULE)
        lines.append(f"FORMATTED EVIDENCE (first {MAX_DISPLAY})")
        if evidence:
            for record in evidence[:MAX_DISPLAY]:
                lines.append(f"  {record}")
        else:
            lines.append("(no evidence)")
        lines.append(RULE)
        lines.append("EXECUTION TIME")
        lines.append(f"  SQL Generation     : {_fmt_seconds(gen_time)}")
        lines.append(f"  SQLite Execution   : {_fmt_seconds(exec_time)}")
        lines.append(f"  Evidence Formatting: {_fmt_seconds(fmt_time)}")
        lines.append(f"  Total Pipeline Time: {_fmt_seconds(total_time)}")
        lines.append("")

    # --- final summary ---
    def _avg(values: list[float]) -> str:
        return f"{sum(values) / len(values):.3f}s" if values else "n/a"

    overall_pass = (
        stats["gen_ok"] == stats["total"]
        and stats["exec_ok"] == stats["total"]
        and stats["fmt_ok"] == stats["total"]
    )

    lines.append(SECTION)
    lines.append("END-TO-END PIPELINE SUMMARY")
    lines.append(SECTION)
    lines.append(f"Total Benchmark Queries       : {stats['total']}")
    lines.append(f"Successful SQL Generations    : {stats['gen_ok']}")
    lines.append(f"Successful SQL Executions     : {stats['exec_ok']}")
    lines.append(f"Successful Evidence Formatting : {stats['fmt_ok']}")
    lines.append(f"Failed SQL Generations        : {stats['gen_fail']}")
    lines.append(f"Failed SQL Executions         : {stats['exec_fail']}")
    lines.append(f"Failed Formatting             : {stats['fmt_fail']}")
    lines.append(f"Average SQL Generation Time   : {_avg(stats['gen_times'])}")
    lines.append(f"Average SQL Execution Time    : {_avg(stats['exec_times'])}")
    lines.append(f"Average Formatting Time       : {_avg(stats['fmt_times'])}")
    lines.append(f"Average End-to-End Time       : {_avg(stats['total_times'])}")
    lines.append(f"Overall Status                : {'PASS' if overall_pass else 'FAIL'}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {OUTPUT_PATH}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
