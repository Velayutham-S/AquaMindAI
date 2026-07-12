"""Evidence Formatter for the AquaMind AI Data Agent.

Single responsibility: convert raw ``sqlite3.Row`` results into structured
evidence (a list of dictionaries), one dictionary per row.

This is a **pure transformation layer**. It does NOT execute SQL, call the LLM,
summarize, generate natural language, sort, aggregate, filter, rank, compute
statistics, rename keys, change data types, infer missing values, or drop NULLs.
Column names and values are preserved exactly as returned by SQLite.

Public interface:
    EvidenceFormatter().format(rows) -> list[dict]
"""

from __future__ import annotations

import sqlite3
from typing import Any


class EvidenceFormatter:
    """Transforms SQLite rows into structured evidence dictionaries."""

    def format(self, rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        """Return one dictionary per row, preserving column names and values.

        An empty input returns an empty list. NULL values are preserved as
        ``None``; data types are left exactly as SQLite returned them.
        """
        if not rows:
            return []
        return [dict(row) for row in rows]
